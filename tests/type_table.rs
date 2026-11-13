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

/// `local_type` (the accessor of the emitter's `var_types`): in `check` mode the walk's declared type of a local
/// is compared with the emitter's, a spelling or a representation (`'inline`, a borrow, `'shared`) aside.
#[test]
fn check_finds_no_declared_type_the_walk_contradicts() {
    let log = std::env::temp_dir().join(format!("boring_type_table_check_{}.log", std::process::id()));
    let _ = std::fs::remove_file(&log);
    for case in ["structs", "modules", "mut_scenarios_2", "trait_scenarios_3", "field_scenarios", "qualifier_infer", "monomorphize_struct", "free_fn_overloading", "method_overloading", "misc_scenarios_6"] {
        let o = Command::new(BIN)
            .args(["build", "--emit-rust", &format!("tests/cases/{case}.br")])
            .env("BORING_TYPE_TABLE", "check")
            .env("BORING_TYPE_TABLE_LOG", &log)
            .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")))
            .output().unwrap();
        assert!(o.status.success(), "{case}: {}", String::from_utf8_lossy(&o.stderr));
    }
    let text = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_file(&log);
    let bad: Vec<&str> = text.lines().filter(|l| l.starts_with("TYPEVARMISMATCH") || l.starts_with("TYPESCALARMISMATCH") || l.starts_with("TYPESTDMISMATCH") || l.starts_with("TYPEINITMISMATCH") || l.starts_with("LASTUSEMISMATCH")).collect();
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// `var_init_exprs` is flat by name and never reset between bodies: a bare field `n` of a method reads the initialiser
/// of a `let n` of another function. `check` reports it (`TYPEINITSTRAY`), walk mode withholds it, same Rust.
#[test]
fn a_stale_initialiser_is_reported_and_withheld() {
    let dir = std::env::temp_dir().join(format!("boring_type_table_stale_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("stale.br");
    std::fs::write(&src, "def warm():\n    let n = 2.0\n    print \"{n}\"\n\nstruct Wave:\n    float32 n\n\n    req float32 half():\n        n / 2\n\nwarm()\nlet w = Wave(3.0)\nprint \"{w.half()}\"\n").unwrap();
    let log = dir.join("check.log");
    let run = |mode: &str| {
        let mut c = Command::new(BIN);
        c.args(["build", "--emit-rust", src.to_str().unwrap()]).env("BORING_TYPE_TABLE", mode)
            .env("BORING_TYPE_TABLE_LOG", &log).current_dir(Path::new(env!("CARGO_MANIFEST_DIR")));
        let o = c.output().unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8_lossy(&o.stdout).into_owned()
    };
    let walked = run("walk");
    assert_eq!(walked, run("off"));
    run("check");
    let text = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(text.contains("TYPEINITSTRAY"), "{text}");
}

/// The representation half of the move-or-clone choice: the emitter's sets and the decisions pass can disagree (a local
/// a `task:` captures is decided `'actor`, while the emitter does not track it behind a lock). `check` reports it, and
/// walk mode, which moves only when both agree, clones where `off` moves.
#[test]
fn a_last_use_the_decisions_do_not_allow_to_move_is_cloned() {
    let dir = std::env::temp_dir().join(format!("boring_type_table_rep_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("rep.br");
    std::fs::write(&src, "struct Bag:\n    [int] xs\n\ndef arr_actor():\n    var mut xs = [1, 2]\n    task:\n        xs.push(3)\n    let b = Bag(xs)\n    print \"{b.xs.len()}\"\n\narr_actor()\n").unwrap();
    let log = dir.join("check.log");
    let run = |mode: &str| {
        let o = Command::new(BIN).args(["build", "--emit-rust", src.to_str().unwrap()])
            .env("BORING_TYPE_TABLE", mode).env("BORING_TYPE_TABLE_LOG", &log)
            .current_dir(Path::new(env!("CARGO_MANIFEST_DIR"))).output().unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8_lossy(&o.stdout).into_owned()
    };
    let walked = run("walk");
    let off = run("off");
    run("check");
    let text = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(text.contains("LASTUSEREPMISMATCH"), "{text}");
    assert!(off.contains("Bag { xs: xs }") && walked.contains("Bag { xs: xs.clone() }"), "{walked}");
}

/// `is_mutex_local` (the accessor of `var_mutex_types`): for locals that are written `'actor`, inferred `'actor`, or not
/// actors at all, the emitter's lock set equals what the decisions pass and the declared types say. The classes this does
/// not cover are in the design note (20.8 to 20.12); over `tests/cases` there is none left.
#[test]
fn check_finds_no_lock_the_decisions_contradict_for_plain_actor_locals() {
    let log = std::env::temp_dir().join(format!("boring_type_table_lock_{}.log", std::process::id()));
    let _ = std::fs::remove_file(&log);
    for case in ["actor_slot_double_read", "view_scenarios_2", "actor_scenarios", "actor_collection_field_ops", "match_subject_reused_in_arm", "qualifier_gaps_enums", "auto_ref_infer", "qualifier_group_local_caller", "qualifier_defaults_chains", "qualifier_infer", "with_stmt"] {
        let o = Command::new(BIN)
            .args(["build", "--emit-rust", &format!("tests/cases/{case}.br")])
            .env("BORING_TYPE_TABLE", "check")
            .env("BORING_TYPE_TABLE_LOG", &log)
            .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")))
            .output().unwrap();
        assert!(o.status.success(), "{case}: {}", String::from_utf8_lossy(&o.stderr));
    }
    let text = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_file(&log);
    let bad: Vec<&str> = text.lines().filter(|l| l.starts_with("LOCKMISSING") || l.starts_with("LOCKEXTRA")).collect();
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// The file's own statements are one body; the decisions pass infers its qualifiers like the others'. A top-level local named
/// like an earlier function's parameter (`items`) and one made `'actor` by a `task:` read the decisions of scope `""`, and
/// `check` finds no difference with the emitter's own inference; the Rust is the same as without the table.
#[test]
fn top_level_statements_get_their_own_decisions() {
    let dir = std::env::temp_dir().join(format!("boring_type_table_top_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("top.br");
    std::fs::write(&src, "struct Item:\n    int n\n\nint total([Item] items):\n    var s = 0\n    for it in items:\n        s += it.n\n    s\n\nstruct Counter:\n    var int v = 0\n    def inc(): v += 1\n\nlet items = [Item(1), Item(2)]\nprint \"{total(items)}\"\n\nmut c = Counter()\ntask:\n    c.inc()\nc.inc()\nprint \"{c.v}\"\n").unwrap();
    let log = dir.join("check.log");
    let run = |mode: &str| {
        let o = Command::new(BIN).args(["build", "--emit-rust", src.to_str().unwrap()])
            .env("BORING_TYPE_TABLE", mode).env("BORING_TYPE_TABLE_LOG", &log).env("BORING_TYPE_TABLE_STATS", "1")
            .current_dir(Path::new(env!("CARGO_MANIFEST_DIR"))).output().unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        (String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
    };
    let (walked, _) = run("walk");
    assert_eq!(walked, run("off").0);
    let (_, err) = run("check");
    let text = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!text.lines().any(|l| l.starts_with("DECISIONMISMATCH")), "{text}");
    assert!(err.contains(" decisions agree, 0 differ") && !err.contains(" 0 decisions agree"), "{err}");
}

/// The lock state of a name depends on where it is read: inside `with c:` the name stands for the unlocked value, after it
/// the lock again, and a name a `match` binds in several arms takes each arm's qualifier. The walk answers by the position of
/// the binding in force (`Binder`), and `check` finds no lock the emitter holds differently.
#[test]
fn the_lock_state_of_a_name_follows_its_binding_by_position() {
    let dir = std::env::temp_dir().join(format!("boring_type_table_pos_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("pos.br");
    std::fs::write(&src, "struct Counter:\n    var int value = 0\n\n    def inc():\n        value += 1\n\ndef f():\n    mut Counter'actor c = Counter()\n    c.inc()\n    with c:\n        c.inc()\n    c.inc()\n    print \"{c.value}\"\n\nf()\n").unwrap();
    let log = dir.join("check.log");
    let o = Command::new(BIN).args(["build", "--emit-rust", src.to_str().unwrap()])
        .env("BORING_TYPE_TABLE", "check").env("BORING_TYPE_TABLE_LOG", &log).env("BORING_TYPE_TABLE_STATS", "1")
        .current_dir(Path::new(env!("CARGO_MANIFEST_DIR"))).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let err = String::from_utf8_lossy(&o.stderr).into_owned();
    let text = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!text.lines().any(|l| l.starts_with("LOCKMISSING") || l.starts_with("LOCKEXTRA")), "{text}");
    assert!(err.contains(" lock reads agree, 0 differ") && !err.contains(" 0 lock reads agree"), "{err}");
}
