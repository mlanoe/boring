// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A top-level `let c = Slot(value = 0)` (generic struct) carried through `__boring_screen_setup`
// must keep its type arguments in the tuple type / `__App` field (was a bare `Slot`: E0107).

use std::fs;
use std::path::PathBuf;
use std::process::Command;

const PROGRAM: &str = r#"
struct Slot<T>:
    var mut T value

def int bump(Slot<int> c):
    c.value

let screen = Screen(Dimension(8, 8), title = "t")
let c = Slot(value = 0)
let f = Slot(value = 0.5)

kernel Fill:
    mut [uint]'surface pixels

    init(Dimension d):
        pixels = [0 for ..<d.width * d.height]

    def ():
        pixels[.thread.x] = 1

var mut k = Fill(Dimension(8, 8))
kernel:
    loop:
        let n = bump(c)
        let g = f.value
        k(block = 8)
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;

#[test]
fn wgpu_screen_setup_keeps_generic_struct_arguments() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_setup_generic_struct");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br = tmp.join("test.br");
    fs::write(&br, PROGRAM).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--target", "wgpu"]).arg(&br).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let rs = fs::read_to_string(tmp.join("test_wgpu/src/main.rs")).unwrap();
    assert!(rs.contains("Slot<isize>") && rs.contains("Slot<f64>"), "{rs}");
    assert!(!rs.contains(": Slot,") && !rs.contains("(Slot,"), "bare generic `Slot` leaked:\n{rs}");
}
