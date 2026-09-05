// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for `boring build --emit-rust <file>` silently dropping every
// module produced by a cross-file/cross-project `use` (a plain local `use <file>`
// resolved against `source_dir`, or a named `[deps]` dependency resolved against
// its own root -- both funnel into `Transpiler::inline_boring_use`, which stashes
// its output in `TranspileOutput::modules` rather than the main `code` string, see
// src/transpiler/emit_top.rs). Only the "write a real Cargo project to disk" path
// (`emit_rust_to_dir`, used by plain `boring build`) ever wrote `t.modules` out --
// one `include!`d `src/<name>.rs` per module. `print_rust` (the `--emit-rust`
// entry point in src/main.rs) printed `out.code` alone and threw the modules away,
// so the call site referencing an imported function/struct compiled fine in
// isolation but the definition it needed never appeared anywhere in the output.
//
// This is exactly the gap `scratch-boring/boring/regen.sh` had to work around
// (running a real project-mode build into a throwaway directory and manually
// inlining its `include!` files) rather than using `--emit-rust` directly.
//
// Each test below feeds `boring build --emit-rust`'s stdout straight to `rustc`
// and runs the resulting binary -- proving the module's code is not just present
// as text, but actually compiles and executes correctly as part of one flat Rust
// stream, which is what a real project-mode `boring build` already did before this
// fix and what `--emit-rust` must now do too.
//
// Run with:
//   cargo test --test emit_rust_modules

use std::path::Path;
use std::process::Command;

/// Runs `boring build --emit-rust <br_file>`, writes stdout to `<out_dir>/gen.rs`,
/// compiles it standalone with `rustc`, runs the resulting binary, and returns its
/// captured stdout. Panics with full context on any failure along the way.
fn emit_and_run(br_file: &Path, out_dir: &Path) -> String {
    let bin = env!("CARGO_BIN_EXE_boring");
    std::fs::create_dir_all(out_dir).expect("failed to create output dir");

    let emit = Command::new(bin)
        .arg("build").arg(br_file)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "boring build --emit-rust failed for {}:\n{}",
        br_file.display(),
        String::from_utf8_lossy(&emit.stderr)
    );
    assert!(
        !emit.stdout.is_empty(),
        "boring build --emit-rust produced empty stdout for {}",
        br_file.display()
    );

    let rs_path = out_dir.join("gen.rs");
    std::fs::write(&rs_path, &emit.stdout).expect("failed to write gen.rs");

    let bin_path = out_dir.join("gen_bin");
    let rustc = Command::new("rustc")
        .arg("--edition").arg("2021")
        .arg(&rs_path)
        .arg("-o").arg(&bin_path)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke rustc: {}", e));
    assert!(
        rustc.status.success(),
        "rustc failed to compile --emit-rust output for {} \
         (a cross-file/cross-project `use` module was likely dropped from stdout):\n\
         --- stderr ---\n{}\n--- generated source ---\n{}",
        br_file.display(),
        String::from_utf8_lossy(&rustc.stderr),
        String::from_utf8_lossy(&emit.stdout),
    );

    let run = Command::new(&bin_path)
        .output()
        .unwrap_or_else(|e| panic!("failed to run compiled binary for {}: {}", br_file.display(), e));
    assert!(
        run.status.success(),
        "compiled binary for {} exited with an error:\n{}",
        br_file.display(),
        String::from_utf8_lossy(&run.stderr)
    );

    String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n")
}

/// Plain local `use <file>` -- no boring.toml, no [deps], just two `.br` files in
/// the same directory (`cross_file_use_main.br` / `cross_file_use_lib.br`). This
/// also exercises `print_rust`'s `source_dir` derivation: before this fix, `--emit-
/// rust`'s standalone-file path never set `source_dir` at all, so the import would
/// have failed to resolve as soon as the test ran from any cwd other than
/// `tests/cases/` (which `cargo test` never does -- the crate root is cwd).
#[test]
fn emit_rust_includes_local_cross_file_use_module() {
    let br_file = Path::new("tests/cases/cross_file_use_main.br");
    let out_dir = Path::new("tests/cases/cross_file_use_emit_rust_rust");

    let actual = emit_and_run(br_file, out_dir);
    assert_eq!(actual.trim_end(), "42");
}

/// Named `[deps]` dependency (`tests/cases/cross_project_dep/boring.toml`'s
/// `[deps] numlib = "../fixtures/dep_numlib"`) -- the exact shape
/// `scratch-boring`'s migration to `boring-numlib` hit. Already covered for
/// project-mode `boring build` by `tests/transpile.rs`'s
/// `transpile_project_test!(cross_project_dep)` and for `boring run` by
/// `tests/run.rs`'s `cross_project_dep` -- this is the missing `--emit-rust` leg.
#[test]
fn emit_rust_includes_deps_cross_project_module() {
    let br_file = Path::new("tests/cases/cross_project_dep/src/main.br");
    let out_dir = Path::new("tests/cases/cross_project_dep_emit_rust_rust");

    let actual = emit_and_run(br_file, out_dir);
    assert_eq!(actual.trim_end(), "21\n42");
}

/// Cross-file monomorphization (extends `emit_rust_includes_local_cross_file_use_module`
/// above): `Wrapper<T>` is declared in `monomorphize_cross_file_lib.br`, and specialized
/// via a turbofish construction site (`Wrapper<string>(...)`) in
/// `monomorphize_cross_file_main.br`, a DIFFERENT file that `use`s it. Proves the
/// specialized `Wrapper_string` clone (emitted once, into the declaring file's own
/// module) and the rewritten call site (a bare, unprefixed identifier — see
/// `src/transpiler/monomorphize.rs`'s module doc comment for why no prefix is needed)
/// both actually land in the flattened `--emit-rust` output and real-compile with rustc,
/// not just under project-mode `boring build` (already covered by
/// `tests/transpile.rs`'s `transpile_test!(monomorphize_cross_file_main)`).
#[test]
fn emit_rust_includes_cross_file_monomorphized_module() {
    let br_file = Path::new("tests/cases/monomorphize_cross_file_main.br");
    let out_dir = Path::new("tests/cases/monomorphize_cross_file_emit_rust_rust");

    let actual = emit_and_run(br_file, out_dir);
    assert_eq!(actual.trim_end(), "hi");
}

/// [deps]-based cross-project monomorphization (extends
/// `emit_rust_includes_deps_cross_project_module` above the same way
/// `emit_rust_includes_cross_file_monomorphized_module` extends
/// `emit_rust_includes_local_cross_file_use_module`): `Wrapper<T>` -- a generic struct
/// with a `type let T` field depending on its own type parameter, per
/// `src/transpiler/monomorphize.rs`'s module doc comment -- is declared in the
/// DEPENDENCY project (`tests/cases/fixtures/dep_monomorphize/src/wrapper.br`), resolved
/// via `tests/cases/cross_project_dep_monomorphize/boring.toml`'s own `[deps]` section,
/// while the concrete turbofish construction site (`Wrapper<string>(...)`) lives in the
/// CONSUMING project's `src/main.br`, a different file in a different `[deps]` project.
/// Proves the specialized `Wrapper_string` clone and the rewritten call site both
/// actually land in the flattened `--emit-rust` output and real-compile with rustc, not
/// just under project-mode `boring build` (already covered by
/// `tests/transpile.rs`'s `transpile_project_test!(cross_project_dep_monomorphize)`) or
/// `boring run` (`tests/run.rs`'s `cross_project_dep_monomorphize`).
#[test]
fn emit_rust_includes_deps_cross_project_monomorphized_module() {
    let br_file = Path::new("tests/cases/cross_project_dep_monomorphize/src/main.br");
    let out_dir = Path::new("tests/cases/cross_project_dep_monomorphize_emit_rust_rust");

    let actual = emit_and_run(br_file, out_dir);
    assert_eq!(actual.trim_end(), "hi\nhi");
}
