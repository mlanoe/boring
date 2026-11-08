// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Type generics in device code on the Metal, CUDA, ROCm and wgpu targets
// (`src/transpiler/device_generic.rs`): explicit instantiations of generic structs and helper
// functions become concrete copies (`Pair_int`, `twice_float32`); everything else generic is a
// Boring error naming the construct. Fixture: `tests/cases/device_generics.br`; the expected
// values are worked out in the comments of `metal_runs_monomorphised_generics`.
//
// Run with: tools/test.sh --test device_generics

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FIXTURE: &str = "tests/cases/device_generics.br";

fn build(source: &str, target: &str, name: &str) -> (PathBuf, Output) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("device_generics_{name}_{target}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("main.br"), source).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", target, "main.br"])
        .current_dir(&dir)
        .output()
        .unwrap();
    (dir, out)
}

const TARGETS: [(&str, &str); 4] = [
    ("metal", "main_metal/kernels/main.metal"),
    ("wgpu", "main_wgpu/shaders/main.wgsl"),
    ("cuda", "main_cuda/kernels/main.cu"),
    ("rocm", "main_rocm/kernels/main.hip"),
];

#[test]
fn generic_structs_and_functions_become_concrete_device_code() {
    let src = std::fs::read_to_string(FIXTURE).unwrap();
    for (target, file) in TARGETS {
        let (dir, out) = build(&src, target, "shape");
        assert!(out.status.success(), "{target}: {}", String::from_utf8_lossy(&out.stderr));
        let device = std::fs::read_to_string(dir.join(file)).unwrap();
        assert!(!device.contains("unsupported"), "{target}: {device}");
        // Instances, and no generic leftover (`T` as a type, the un-mangled helper).
        for needle in ["twice_int", "twice_float32", "mid_int", "swap_int", "Pair_int__sum", "Pair_float32__sum"] {
            assert!(device.contains(needle), "{target}: missing {needle}: {device}");
        }
        for stale in [" twice(", " T twice", "(T x)", "Pair__"] {
            assert!(!device.contains(stale), "{target}: leftover {stale}: {device}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_runs_monomorphised_generics() {
    let src = std::fs::read_to_string(FIXTURE).unwrap();
    let (dir, out) = build(&src, "metal", "run");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(dir.join("main_metal/Cargo.toml"))
        .arg("--target-dir")
        .arg(dir.join("target"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&run.stdout).trim().to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    // Thread i: q = Pair<int>(i, 3); sum 3+i; mid = 2i + 3; swap(q).a = 3; bias 100; twice(i) = 2i.
    // Total = (3+i) + (2i+3) + 3 + 100 + 2i = 109 + 5i. fout: 1.5 + 2.0 + 2 * 0.25 = 4.
    assert_eq!(stdout, "109 114 119 124\n4 4");
}

const PRELUDE: &str = "struct Pair<T>:
    T a
    T b

T twice<T>(T x):
    x + x

mut [int]'kernel'unified out = [0 for ..<4]
kernel O:
    mut [int]'unified out
    init(mut [int]'unified o):
        out = o
    def ():
        let i = .thread.x
";
const EPILOGUE: &str = "
mut o = O(out)
kernel:
    o(block = 4)
";

#[test]
fn unsupported_generic_uses_are_boring_errors_on_every_target() {
    let cases: [(&str, String, &str); 5] = [
        ("bare_struct", format!("{PRELUDE}        let p = Pair(a = i, b = 3)\n        out[i] = p.a\n{EPILOGUE}"), "generic struct `Pair` is built in device code without type arguments"),
        ("bare_fn", format!("{PRELUDE}        out[i] = twice(i)\n{EPILOGUE}"), "generic function `twice` is called from device code without type arguments"),
        ("bare_type", format!("{PRELUDE}        let Pair q = Pair<int>(a = i, b = 3)\n        out[i] = q.a\n{EPILOGUE}"), "local `q` is typed with the generic struct `Pair`"),
        ("arity", format!("{PRELUDE}        out[i] = twice<int, int>(i)\n{EPILOGUE}"), "`twice` takes 1 type argument(s)"),
        (
            "kernel",
            "mut [int]'kernel'unified out = [0 for ..<4]\nkernel K<T>:\n    mut [T]'unified buf\n    init(mut [T]'unified b):\n        buf = b\n    def ():\n        buf[.thread.x] = 1\n\nmut k = K<int>(out)\nkernel:\n    k(block = 4)\n".to_string(),
            "generic kernel `K`",
        ),
    ];
    for (name, src, message) in &cases {
        for (target, _) in TARGETS {
            let (dir, out) = build(src, target, name);
            let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
            let _ = std::fs::remove_dir_all(&dir);
            assert!(!out.status.success(), "{name}/{target}: should be rejected");
            assert!(stderr.contains(message), "{name}/{target}: {stderr}");
        }
    }
}

// An untyped-field struct (D6) instantiated explicitly is an ordinary generic struct; a bare
// construction is still rejected (see `device_structs.rs`).
#[test]
fn explicitly_instantiated_implicit_struct_is_accepted() {
    let src = "struct Wrap:\n    v\n    int n\n\nmut [int]'kernel'unified out = [0 for ..<4]\nkernel O:\n    mut [int]'unified out\n    init(mut [int]'unified o):\n        out = o\n    def ():\n        let i = .thread.x\n        let w = Wrap<int>(v = i, n = 2)\n        out[i] = w.v + w.n\n\nmut o = O(out)\nkernel:\n    o(block = 4)\n";
    for (target, file) in TARGETS {
        let (dir, out) = build(src, target, "implicit");
        assert!(out.status.success(), "{target}: {}", String::from_utf8_lossy(&out.stderr));
        let device = std::fs::read_to_string(dir.join(file)).unwrap();
        assert!(device.contains("w__v"), "{target}: {device}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
