// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Implicit generics phase 1 across files: a `def` with no return type that returns a struct
// declared in a file it `use`s gets that struct's type. `parser::parse` sees one file only, so the
// pass is re-run from `desugar_array_block_with_imports` with the imported items. Before, the
// return type silently stayed `void` (`cannot access field 'payload' on Void` / rustc E0308).
//
// Run with: tools/test.sh --test infer_return_types_across_files

use std::path::{Path, PathBuf};
use std::process::Command;

fn project(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("infer_return_types_across_files").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    for (file, src) in files {
        let p = dir.join(file);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, src).unwrap();
    }
    dir
}

const TYPES: &str = "struct Resp:\n    let status\n    let payload\n";
const MAIN: &str = "use types\n\ndef mk():\n    return Resp(status = 200, payload = \"hi\")\n\ndef main():\n    let r = mk()\n    print r.payload\n";

#[test]
fn run_infers_return_type_of_imported_struct() {
    let dir = project("run", &[("types.br", TYPES), ("main.br", MAIN)]);
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).current_dir(&dir).args(["run", "main.br"]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hi");
}

#[test]
fn build_emits_inferred_return_type_of_imported_struct() {
    let dir = project("build", &[("types.br", TYPES), ("main.br", MAIN)]);
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).current_dir(&dir).args(["build", "--emit-rust", "main.br"]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let rust = String::from_utf8_lossy(&out.stdout);
    assert!(rust.contains("fn mk() -> Resp<isize, Arc<str>>"), "{rust}");
}

#[test]
fn run_infers_through_imported_function() {
    let types = format!("{TYPES}\ndef base():\n    return Resp(status = 1, payload = \"x\")\n");
    let main = "use types\n\ndef again():\n    return base()\n\ndef main():\n    print again().payload\n";
    let dir = project("fn", &[("types.br", &types), ("main.br", main)]);
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).current_dir(&dir).args(["run", "main.br"]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "x");
}

fn run_err(name: &str, main: &str) -> String {
    let dir = project(name, &[("types.br", TYPES), ("main.br", main)]);
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).current_dir(&dir).args(["run", "main.br"]).output().unwrap();
    assert!(!out.status.success());
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn untyped_field_given_an_empty_collection_is_an_error_for_an_imported_struct() {
    let err = run_err("empty", "use types\n\ndef main():\n    let r = Resp(status = 1, payload = [])\n    print r.status\n");
    assert!(err.contains("cannot infer the type of `Resp.payload`"), "{err}");
    assert!(err.contains("main.br"), "{err}");
}

#[test]
fn array_mixing_two_instances_of_an_imported_implicit_struct_is_an_error() {
    let err = run_err("mixed", "use types\n\ndef a():\n    return Resp(status = 1, payload = \"s\")\n\ndef b():\n    return Resp(status = 1, payload = 2)\n\ndef main():\n    let xs = [a(), b()]\n    print xs.len\n");
    assert!(err.contains("an array holds one type"), "{err}");
}
