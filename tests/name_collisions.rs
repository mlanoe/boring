// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression tests: user-chosen names must not silently collide with other namespaces.
//
// 1. `boring run`: a user `def channel(...)` was shadowed by the interpreter's special-cased
//    `channel(...)` builtin (it returned a (Sender, Receiver) tuple instead of calling the
//    user's function). Same for `timeout`. User definitions now win.
// 2. `boring build`: a user `struct Box` collides with `std::boxed::Box` in the generated Rust
//    (E0107 wherever the generated code writes `Box<dyn ...>`). Now a clear diagnostic.
// 3. Overload resolution: `mut [float]` params (`Type::Mut`) and array-literal arguments
//    (`[1.0]` was typed as `[int]`) never matched, so the call silently fell back to the FIRST
//    declared overload of the name — wherever it came from, including a same-named function
//    in another module — producing `/* missing arg */` calls.
//
// Run with:
//   cargo test --test name_collisions

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn project(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    for (file, src) in files {
        let p = dir.join(file);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, src).unwrap();
    }
    dir
}

fn boring(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_boring"))
        .current_dir(dir)
        .args(args)
        .output()
        .expect("failed to invoke boring")
}

#[test]
fn user_def_named_like_interpreter_builtin_wins() {
    let dir = project("name_collision_builtin", &[("main.br", "\
def int channel(uint p, int shift):
    return int((p >> shift) & 255)

def int timeout(int a, int b):
    return a + b

def main():
    let x = channel(0x00FF00, 8)
    print(x * 2.0)
    print(timeout(1, 2))
")]);
    let out = boring(&dir, &["run", "main.br"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(stdout.trim(), "510\n3");
}

#[test]
fn user_struct_named_box_with_trait_objects_is_a_clear_error() {
    let dir = project("name_collision_box", &[("main.br", "\
struct Box:
    int w
    int h

def int risky(int n) throws:
    if n < 0:
        throw \"neg\"
    return n

def main() throws:
    let b = Box(w: 2, h: 3)
    print b.w + risky(b.h)
")]);
    let out = boring(&dir, &["build", "main.br", "--emit-rust"]);
    assert!(!out.status.success(), "expected a diagnostic for `struct Box`");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("`Box` collides with Rust's `std::boxed::Box`"), "stderr:\n{}", stderr);
}

#[test]
fn overload_with_mut_array_params_not_hijacked_by_other_modules_same_name() {
    let dir = project("name_collision_overload", &[
        ("boring.toml", "[project]\nname = \"p\"\nversion = \"0.1.0\"\n[deps]\ndep = \"dep\"\n"),
        ("dep/src/lib.br", "\
def float fill(mut [float] cur, [float] target, [int] group, float remaining):
    cur[0] = remaining
    return remaining

pub def float run_fill():
    var cur = [0.0, 0.0]
    return fill(cur, [1.0], [1], 2.0)
"),
        ("main.br", "\
use dep.lib.*

def string fill(float x, float y, float w, float h, string s):
    return s

def main():
    print run_fill()
    print fill(1.0, 1.0, 1.0, 1.0, \"a\")
"),
    ]);
    let out = boring(&dir, &["build", "--emit-rust"]);
    let code = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(!code.contains("missing arg"), "call was bound to the wrong overload:\n{}", code);
    assert!(code.contains("return fill__arr_float__arr_float__arr_int__float(&mut cur"), "{}", code);
}
