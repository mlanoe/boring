// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression tests for two `[external_fns]` ergonomic gaps (builder-pattern external crate, modelled
// on swash's `Render`):
//   1. a chained call on a temporary (`Render.new(s).format(f).render(sc, g)`) must resolve each
//      later receiver's type so the borrow table applies (`render` takes `&mut Scaler`);
//   2. a list literal passed to a `&[T]` slot (`Render.new([Source.Outline])`) must be bound to a
//      hidden local, not emitted as a temporary `&vec![..]` dropped while the result borrows it.
// The generated project is built against a small local path crate, so no network is needed.

use std::path::Path;
use std::process::Command;

const CRATE_LIB: &str = r#"
#[derive(Clone, Copy)]
pub enum Source { Outline }
pub enum Format { Alpha }
pub struct Scaler { pub calls: u32 }
impl Scaler { pub fn new(calls: u32) -> Self { Scaler { calls } } }
pub struct Render<'a> { sources: &'a [Source], alpha: bool }
impl<'a> Render<'a> {
    pub fn new(sources: &'a [Source]) -> Self { Render { sources, alpha: false } }
    pub fn format(&mut self, f: Format) -> &mut Self { self.alpha = matches!(f, Format::Alpha); self }
    pub fn render(&mut self, scaler: &mut Scaler, glyph: u16) -> u32 {
        scaler.calls += 1;
        glyph as u32 + self.sources.len() as u32 + self.alpha as u32
    }
}
"#;

const MAIN_BR: &str = "use bld.Render
use bld.Source
use bld.Format
use bld.Scaler

def main():
    mut Scaler scaler = Scaler(0)
    let img = Render.new([Source.Outline]).format(Format.Alpha).render(scaler, 7)
    print \"{img}\"
";

fn setup(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("boring_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("bld/src")).unwrap();
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(dir.join("bld/Cargo.toml"),
        "[package]\nname = \"bld\"\nversion = \"0.1.0\"\nedition = \"2021\"\n").unwrap();
    std::fs::write(dir.join("bld/src/lib.rs"), CRATE_LIB).unwrap();
    std::fs::write(dir.join("src/main.br"), MAIN_BR).unwrap();
    std::fs::write(dir.join("boring.toml"), format!(
        "[project]\nname = \"chain\"\nversion = \"0.1.0\"\nmain = \"src/main.br\"\n\n\
         [dependencies]\nbld = {{ path = \"{}\" }}\n\n\
         [external_fns]\n\"Render::new\" = [\"&\"]\n\"Render::render\" = [\"&mut\", \"\"]\n",
        dir.join("bld").display())).unwrap();
    dir
}

fn boring(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_boring")).current_dir(dir).args(args).output().unwrap()
}

#[test]
fn chained_external_call_applies_borrow_table_and_hoists_list_literal() {
    let dir = setup("ext_chain_text");
    let out = boring(&dir, &["build", "--emit-rust", "src/main.br"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let rs = String::from_utf8_lossy(&out.stdout);
    assert!(rs.contains("render(&mut scaler, 7)"), "chained receiver not resolved:\n{rs}");
    assert!(!rs.contains("&vec!["), "temporary borrowed list literal:\n{rs}");
    assert!(rs.contains("Render::new(&__ext_"), "list literal not hoisted:\n{rs}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn chained_external_call_compiles_and_runs() {
    let dir = setup("ext_chain_build");
    let out = boring(&dir, &["build", "src/main.br"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let project = std::fs::read_dir(dir.join("src")).unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.join("Cargo.toml").is_file())
        .expect("no generated Cargo project");
    let run = Command::new("cargo").current_dir(&project)
        .args(["run", "--offline", "--quiet"]).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "9");
    let _ = std::fs::remove_dir_all(&dir);
}
