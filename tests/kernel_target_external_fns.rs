//! `boring build --target wgpu|simd` must honor `boring.toml`'s `[external_fns]` (borrow forms
//! for external crate calls), same as project-mode `boring build` does.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn build(target: &str) -> String {
    let bin = env!("CARGO_BIN_EXE_boring");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("kernel_target_external_fns").join(target);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    fs::write(tmp.join("boring.toml"), "[project]\nname = \"t\"\nversion = \"0.1.0\"\nmain = \"main.br\"\n\n[external_fns]\n\"Shaper::shape\" = [\"&mut\", \"&\"]\n").unwrap();
    fs::write(tmp.join("main.br"), "kernel Noop:\n    def run():\n        pass\n\nint main():\n    mut s = Shaper.new()\n    let words = [1, 2]\n    Shaper.shape(s, words)\n    0\n").unwrap();
    let out = Command::new(bin).args(["build", "--target", target, "main.br"]).current_dir(&tmp).output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let rs = fs::read_to_string(tmp.join(format!("main_{target}/src/main.rs")))
        .unwrap_or_else(|_| panic!("no generated main.rs; stderr:\n{stderr}"));
    rs
}

#[test]
fn wgpu_honors_external_fns_borrows() {
    let rs = build("wgpu");
    assert!(rs.contains("Shaper::shape(&mut s, &words)"), "borrows missing:\n{rs}");
}

#[test]
fn simd_honors_external_fns_borrows() {
    let rs = build("simd");
    assert!(rs.contains("Shaper::shape(&mut s, &words)"), "borrows missing:\n{rs}");
}
