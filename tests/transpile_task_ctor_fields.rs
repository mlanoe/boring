// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A positional constructor argument for a `'actor'task` / `'guard'task` struct field must be
// wrapped exactly like the labeled form (`T(stats = Counter())`). Emission-only: the field type
// of a `'task` field vs. its constructor lock flavour is a separate, known gap.
//
// Run with:
//   cargo test --test transpile_task_ctor_fields

use std::process::Command;

fn ctor_lines(src: &str, name: &str) -> Vec<String> {
    let dir = std::env::temp_dir().join(format!("boring_task_ctor_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("t.br");
    std::fs::write(&file, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--emit-rust"]).arg(&file).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).lines()
        .filter(|l| l.trim_start().starts_with("let ") && l.contains("stats:"))
        .map(|l| l.split_once("= ").unwrap().1.to_string())
        .collect()
}

#[test]
fn positional_matches_labeled_for_task_qualifiers() {
    for (q, lock) in [("actor'task", "Mutex"), ("guard'task", "RwLock")] {
        let src = format!(
            "struct Counter:\n    var int n = 0\nstruct T:\n    mut Counter'{q} stats\n\
let a = T(Counter())\nlet b = T(stats = Counter())\nlet c = T(Counter(3))\nlet d = T(stats = Counter(3))\n");
        let l = ctor_lines(&src, q.replace('\'', "_").as_str());
        assert_eq!(l.len(), 4, "{l:?}");
        assert_eq!(l[0], l[1]);
        assert_eq!(l[2], l[3]);
        assert!(l[0].contains(&format!("{lock}::new(")), "{q}: {}", l[0]);
    }
}
