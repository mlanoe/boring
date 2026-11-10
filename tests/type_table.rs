// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `TypeTable` (src/transpiler/type_table.rs): a first emission records what the emitter answers about
// the struct a variable or field expression resolves to, a second one replays the answers instead of
// consulting its own per-local tables. `BORING_TYPE_TABLE=replay` must not change the Rust.

use std::path::Path;
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_boring");

fn emit(case: &str, table: Option<&str>) -> String {
    let mut cmd = Command::new(BIN);
    cmd.args(["build", "--emit-rust", &format!("tests/cases/{case}.br")])
        .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")));
    if let Some(t) = table { cmd.env("BORING_TYPE_TABLE", t); }
    let o = cmd.output().unwrap();
    assert!(o.status.success(), "{case}: {}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).into_owned()
}

#[test]
fn replaying_or_walking_gives_the_same_rust() {
    for case in [
        "structs", "modules", "mut_scenarios_2", "trait_scenarios_3", "actor_collection_field_ops",
        "implicit_generics_specialise_methods", "implicit_generics_specialise_mod", "field_scenarios",
        "qualifier_infer", "monomorphize_struct",
    ] {
        let plain = emit(case, None);
        assert_eq!(plain, emit(case, Some("replay")), "{case} (replay)");
        assert_eq!(plain, emit(case, Some("walk")), "{case} (walk)");
    }
}

#[test]
fn stats_report_the_table_and_what_a_walk_of_its_own_gets() {
    let o = Command::new(BIN)
        .args(["build", "--emit-rust", "tests/cases/structs.br"])
        .env("BORING_TYPE_TABLE", "stats")
        .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")))
        .output().unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("type table: ") && err.contains("type table built alone: agree "), "{err}");
}
