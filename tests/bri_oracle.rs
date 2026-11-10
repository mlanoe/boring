// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The `.bri` oracle (docs/design-notes/typed-specialisation.md, sections 3.6 and 11).
//
// A `.bri` is valid Boring, so for every `tests/cases/*.br`:
//   1. `boring run x.br` and `boring run x.bri` print the same output (level 1, `mono`);
//   2. `boring build --emit-rust x.br` and `... x.bri` generate the same Rust (level 2).
// The `full` level is not an oracle (a written qualifier is evidence, an inferred one is not): its
// mismatch count is measured by the ignored test `bri_full_roundtrip_report`.
//
// The `.bri` files are written into a scratch copy of the cases directory (siblings are found by
// `use`), never into `tests/cases`.

use std::path::{Path, PathBuf};
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_boring");

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let (from, to) = (entry.path(), dst.join(entry.file_name()));
        if from.is_dir() {
            copy_dir(&from, &to);
        } else if let Err(e) = std::fs::copy(&from, &to) {
            // a dangling symlink in the fixtures is not our business
            eprintln!("skip {}: {e}", from.display());
        }
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    copy_dir(Path::new("tests/cases"), &dir);
    dir
}

struct Out {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run(dir: &Path, args: &[&str]) -> Out {
    let o = Command::new(BIN).args(args).current_dir(dir).stdin(std::process::Stdio::null()).output().unwrap();
    Out {
        ok: o.status.success(),
        stdout: String::from_utf8_lossy(&o.stdout).replace("\r\n", "\n"),
        stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

/// `__stream_<source line>` names depend on the line of the `for` in the file, which a reprint changes.
fn normalise(rust: &str) -> String {
    // Compiler-made names (`__stream_<line>`, `__comp_<line>_<col>`, `__a_axis0_<line>_<col>`) carry source
    // positions, which the printed file moves: their digit runs are blanked.
    let bytes = rust.as_bytes();
    let mut out = String::with_capacity(rust.len());
    let mut i = 0;
    while i < bytes.len() {
        if rust[i..].starts_with("__") && (i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_')) {
            let end = i + bytes[i..].iter().take_while(|b| b.is_ascii_alphanumeric() || **b == b'_').count();
            let ident = &rust[i..end];
            let ib = ident.as_bytes();
            let mut prev_us = false;
            let mut j = 0;
            while j < ib.len() {
                if prev_us && ib[j].is_ascii_digit() {
                    let n = ib[j..].iter().take_while(|b| b.is_ascii_digit()).count();
                    out.push('N');
                    j += n;
                    prev_us = false;
                    continue;
                }
                out.push(ib[j] as char);
                prev_us = ib[j] == b'_';
                j += 1;
            }
            i = end;
        } else {
            let ch = rust[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

fn cases(dir: &Path, need_expected: bool) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir).unwrap()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".br").map(|s| s.to_string())))
        .filter(|n| !need_expected || dir.join(format!("{n}.expected")).exists())
        .collect();
    names.sort();
    names
}

/// Runs `f` over `names` on all cores; the results come back in name order.
fn par_map<T: Send>(names: &[String], f: impl Fn(&str) -> T + Sync) -> Vec<(String, T)> {
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
    let chunk = names.len().div_ceil(workers).max(1);
    let mut out = Vec::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = names.chunks(chunk).map(|c| {
            let f = &f;
            s.spawn(move || c.iter().map(|n| (n.clone(), f(n))).collect::<Vec<_>>())
        }).collect();
        for h in handles { out.extend(h.join().unwrap()); }
    });
    out
}

enum Verdict { Ok, Skipped, Mismatch(String) }

fn write_bri(dir: &Path, name: &str, level: &str) -> Result<(), String> {
    let printed = run(dir, &["check", &format!("--emit-bri={level}"), &format!("{name}.br")]);
    if !printed.ok {
        return Err(format!("print failed: {}", printed.stderr.lines().next().unwrap_or("")));
    }
    std::fs::write(dir.join(format!("{name}.bri")), printed.stdout).unwrap();
    Ok(())
}

fn run_oracle(dir: &Path, name: &str, level: &str) -> Verdict {
    let original = run(dir, &[&format!("{name}.br")]);
    if !original.ok { return Verdict::Skipped; }
    if let Err(e) = write_bri(dir, name, level) { return Verdict::Mismatch(e); }
    let again = run(dir, &[&format!("{name}.bri")]);
    if !again.ok {
        return Verdict::Mismatch(format!("the .bri does not run: {}", again.stderr.lines().find(|l| l.contains("error")).unwrap_or("")));
    }
    if original.stdout.trim_end() != again.stdout.trim_end() {
        return Verdict::Mismatch("output differs".into());
    }
    Verdict::Ok
}

/// The top-level items of generated Rust (blocks closed at brace depth 0), sorted.
fn sorted_items(rust: &str) -> Vec<String> {
    let mut items: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut depth: i64 = 0;
    for line in rust.lines() {
        cur.push_str(line);
        cur.push('\n');
        for c in line.chars() {
            match c { '{' => depth += 1, '}' => depth -= 1, _ => {} }
        }
        if depth <= 0 && (line.trim_end().ends_with('}') || line.trim_end().ends_with(';')) {
            items.push(std::mem::take(&mut cur).trim().to_string());
            depth = 0;
        }
    }
    if !cur.trim().is_empty() { items.push(cur.trim().to_string()); }
    items.retain(|i| !i.trim().is_empty());
    items.sort();
    items
}

fn build_oracle(dir: &Path, name: &str, level: &str) -> Verdict {
    let original = run(dir, &["build", "--emit-rust", &format!("{name}.br")]);
    if !original.ok { return Verdict::Skipped; }
    if let Err(e) = write_bri(dir, name, level) { return Verdict::Mismatch(e); }
    let again = run(dir, &["build", "--emit-rust", &format!("{name}.bri")]);
    if !again.ok {
        return Verdict::Mismatch(format!("the .bri does not build: {}", again.stderr.lines().find(|l| l.contains("error")).unwrap_or("")));
    }
    if normalise(&original.stdout) == normalise(&again.stdout) { return Verdict::Ok; }
    // The same items in another order is the same program: the monomorphizer's instances go at the end of the
    // generated file, but a `.bri` needs them before the statements that use them.
    if sorted_items(&normalise(&original.stdout)) == sorted_items(&normalise(&again.stdout)) { return Verdict::Ok; }
    // A few generated orders come from a hash map (`supertraits`): the source alone has more than one
    // output. Same output set for both is the identity.
    let outputs = |file: String, first: String| -> std::collections::HashSet<String> {
        let mut set = std::collections::HashSet::from([first]);
        for _ in 0..6 { set.insert(normalise(&run(dir, &["build", "--emit-rust", &file]).stdout)); }
        set
    };
    let from_source = outputs(format!("{name}.br"), normalise(&original.stdout));
    let from_bri = outputs(format!("{name}.bri"), normalise(&again.stdout));
    if !from_source.is_disjoint(&from_bri) { return Verdict::Ok; }
    Verdict::Mismatch("generated Rust differs".into())
}

fn summarise(label: &str, results: &[(String, Verdict)]) -> (usize, usize, Vec<String>) {
    let ok = results.iter().filter(|(_, v)| matches!(v, Verdict::Ok)).count();
    let skipped = results.iter().filter(|(_, v)| matches!(v, Verdict::Skipped)).count();
    let bad: Vec<String> = results.iter().filter_map(|(n, v)| match v {
        Verdict::Mismatch(m) => Some(format!("{n}: {m}")),
        _ => None,
    }).collect();
    eprintln!("{label}: {ok} identical, {} different, {skipped} skipped", bad.len());
    (ok, skipped, bad)
}

#[test]
fn mono_bri_runs_like_the_source() {
    let dir = scratch("bri_oracle_run");
    let names = cases(&dir, true);
    let results = par_map(&names, |n| run_oracle(&dir, n, "mono"));
    let (ok, _, bad) = summarise("mono run oracle", &results);
    assert!(ok >= 400, "oracle looks broken: only {ok} cases compared");
    assert!(bad.is_empty(), "{} case(s) differ between x.br and x.bri:\n{}", bad.len(), bad.join("\n"));
}

#[test]
fn mono_bri_builds_the_same_rust() {
    let dir = scratch("bri_oracle_build");
    let names = cases(&dir, false);
    let results = par_map(&names, |n| build_oracle(&dir, n, "mono"));
    let (ok, _, bad) = summarise("mono build oracle", &results);
    assert!(ok >= 400, "oracle looks broken: only {ok} cases compared");
    assert!(bad.is_empty(), "{} case(s) generate different Rust from x.br and x.bri:\n{}", bad.len(), bad.join("\n"));
}

/// Not an oracle: how many cases change when the inferred qualifiers of `full` are read back as
/// written ones. Run with `cargo test --test bri_oracle -- --ignored --nocapture`.
#[test]
#[ignore]
fn bri_full_roundtrip_report() {
    let dir = scratch("bri_oracle_full");
    let run_names = cases(&dir, true);
    let run_results = par_map(&run_names, |n| run_oracle(&dir, n, "full"));
    let (_, _, run_bad) = summarise("full run", &run_results);
    let build_names = cases(&dir, false);
    let build_results = par_map(&build_names, |n| build_oracle(&dir, n, "full"));
    let (_, _, build_bad) = summarise("full build", &build_results);
    eprintln!("full, run mismatches:\n{}", run_bad.join("\n"));
    eprintln!("full, build mismatches:\n{}", build_bad.join("\n"));
}
