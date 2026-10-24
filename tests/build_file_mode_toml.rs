// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `boring build <file>.br` (file argument) must honour the same
// `boring.toml` tables as project mode -- `[dependencies]` (emitted into Cargo.toml) and
// `[external_types]` (affecting the emitted Rust) -- found next to or above the file.

use std::path::Path;
use std::process::Command;

fn boring_build_file(dir: &Path, file: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_boring"))
        .current_dir(dir)
        .args(["build", file])
        .output()
        .expect("failed to invoke boring")
}

#[test]
fn file_arg_build_applies_toml_dependencies_and_external_types() {
    let fixture = std::fs::canonicalize("tests/cases/fixtures/ext_tuple").unwrap();
    let dir = std::env::temp_dir().join(format!("boring_file_mode_toml_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::copy("tests/cases/ext_types_toml_project/src/main.br", dir.join("src/main.br")).unwrap();
    std::fs::write(
        dir.join("boring.toml"),
        format!(
            "[project]\nname = \"fm\"\nversion = \"0.1.0\"\nmain = \"src/main.br\"\n\n\
             [dependencies]\next_tuple = {{ path = \"{}\" }}\n\n\
             [external_types]\ntuple_structs = [\"ExtOpaque\"]\n",
            fixture.display()
        ),
    )
    .unwrap();

    let out = boring_build_file(&dir, "src/main.br");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    let project = std::fs::read_dir(dir.join("src"))
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.join("Cargo.toml").is_file())
        .expect("no generated Cargo project next to the source file");
    let cargo = std::fs::read_to_string(project.join("Cargo.toml")).unwrap();
    assert!(cargo.contains("ext_tuple"), "[dependencies] ignored in file mode:\n{cargo}");
    let rs = std::fs::read_to_string(project.join("src/main.rs")).unwrap();
    assert!(!rs.contains("ExtOpaque::new("), "[external_types] ignored in file mode:\n{rs}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn file_arg_metal_target_applies_toml_dependencies() {
    let dir = std::env::temp_dir().join(format!("boring_file_mode_metal_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("src/main.br"), "def main():\n    print \"hi\"\n").unwrap();
    std::fs::write(
        dir.join("boring.toml"),
        "[project]\nname = \"fm\"\nversion = \"0.1.0\"\nmain = \"src/main.br\"\n\n[dependencies]\nswash = \"0.2.10\"\n",
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .current_dir(&dir)
        .args(["build", "--target", "metal", "src/main.br"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let cargo = std::fs::read_to_string(dir.join("src/main_metal/Cargo.toml")).unwrap();
    assert!(cargo.contains("swash"), "[dependencies] ignored on --target metal:\n{cargo}");
    let _ = std::fs::remove_dir_all(&dir);
}
