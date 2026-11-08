// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `boring <subcommand> --help` / `-h` must print usage on stdout and exit 0, naming every flag the
// subcommand's parser accepts. The flag list is extracted from the parser source in `src/main.rs`
// (the `"--flag" =>` match arms), so a new flag without help text fails here.

use std::process::Command;

fn parser_flags(src: &str, fn_name: &str) -> Vec<String> {
    let start = src.find(&format!("fn {fn_name}")).unwrap_or_else(|| panic!("no fn {fn_name}"));
    let body = &src[start..];
    let end = body[3..].find("\nfn ").map(|i| i + 3).unwrap_or(body.len());
    let mut flags = Vec::new();
    for line in body[..end].lines() {
        let t = line.trim_start();
        if !t.starts_with("\"--") || t.starts_with("\"--\"") { continue; }
        if !t.contains("=>") { continue; }
        let name = t[1..].split('"').next().unwrap().to_string();
        if !flags.contains(&name) { flags.push(name); }
    }
    flags
}

fn help(sub: &str, flag: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).args([sub, flag]).output().unwrap();
    assert!(out.status.success(), "boring {sub} {flag}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn subcommand_help_lists_parser_flags() {
    let src = std::fs::read_to_string("src/main.rs").unwrap();
    let cases: [(&str, Vec<String>); 2] = [
        ("build", parser_flags(&src, "parse_build_command")),
        ("run", parser_flags(&src, "parse_run_flags")),
    ];
    for (sub, flags) in cases {
        assert!(flags.len() >= 5, "{sub}: flag extraction looks broken: {flags:?}");
        for flag_arg in ["--help", "-h"] {
            let text = help(sub, flag_arg);
            assert!(text.contains("Usage: boring"), "{sub}: {text}");
            for f in &flags {
                assert!(text.contains(f.as_str()), "`boring {sub} {flag_arg}` does not mention {f}");
            }
        }
    }
}

#[test]
fn other_subcommands_accept_help() {
    for (sub, needle) in [("check", "--show-types"), ("api", "--check"), ("update", "Usage"), ("new", "Usage")] {
        for flag in ["--help", "-h"] {
            assert!(help(sub, flag).contains(needle), "{sub} {flag}");
        }
    }
}
