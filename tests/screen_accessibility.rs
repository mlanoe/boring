// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `Screen` accessibility (docs/gpu-display.md, "Accessibility"): `screen.setAccessibility(nodes)` and
// `screen.accessibilityRequests`. wgpu bridges them to AccessKit (winit 0.30); simd, Metal, CUDA and ROCm
// ignore the API (the statements are dropped). Generated crates are `cargo check`ed (Metal on macOS only,
// CUDA with `tools/fake-nvcc`, ROCm with a stub build script). No window is opened.

mod common;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// A three-node tree (a group holding a label and a button) sent every frame, and the requests
/// the screen reader sent handed to an ordinary function.
const PROGRAM: &str = r#"
let screen = Screen(Dimension(64, 64), title = "ax", redraw = "demand")

pub struct AxNode:
    pub int id
    pub int parent
    pub string role
    pub string name
    pub string value
    pub string hint
    pub string marked
    pub int level
    pub float x
    pub float y
    pub float w
    pub float h
    pub bool focused
    pub bool disabled
    pub string actions
    pub [int] children

def [AxNode] threeNodes(int frame):
    [
        AxNode(id = 1, parent = 0, role = "group", name = "Form", value = "", hint = "", marked = "", level = 0, x = 0.0, y = 0.0, w = 64.0, h = 64.0, focused = false, disabled = false, actions = "", children = [2, 3]),
        AxNode(id = 2, parent = 1, role = "staticText", name = "Hello", value = "", hint = "", marked = "", level = 0, x = 4.0, y = 4.0, w = 40.0, h = 12.0, focused = false, disabled = false, actions = "", children = []),
        AxNode(id = 3, parent = 1, role = "button", name = "Press {frame}", value = "", hint = "Presses", marked = "", level = 0, x = 4.0, y = 20.0, w = 40.0, h = 16.0, focused = true, disabled = false, actions = "focus,activate", children = [])
    ]

def note(int id, string action, string arg):
    print "{id} {action} {arg}"

kernel Fill:
    mut [uint]'surface pixels
    let Dimension dim

    init(Dimension d):
        pixels = [0xFF101010 for ..<d.width * d.height]
        dim = d

    def ():
        let col = .block.x * .block_dim.x + .thread.x
        let row = .block.y * .block_dim.y + .thread.y
        if col < dim.width and row < dim.height:
            pixels[row * dim.width + col] = 0xFF202020

var mut k = Fill(Dimension(64, 64))

kernel:
    loop:
        screen.setAccessibility(threeNodes(screen.frame))
        for r in screen.accessibilityRequests:
            note(r.id, r.action, r.arg)
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;

fn build(target: &str, name: &str, src: &str) -> (PathBuf, String) {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_accessibility").join(format!("{name}_{target}"));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br = tmp.join("test.br");
    fs::write(&br, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", target])
        .arg(&br)
        .output()
        .unwrap();
    assert!(out.status.success(), "[{name}/{target}] boring build failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let project = tmp.join(format!("test_{target}"));
    let rs = fs::read_to_string(project.join("src/main.rs")).unwrap();
    (project, rs)
}

fn cargo_check(project: &std::path::Path, target: &str) {
    if target == "rocm" {
        // The generated build script compiles the kernel with hipcc and probes the device;
        // neither is needed to type-check the host code.
        fs::write(project.join("build.rs"), r#"fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    std::fs::write(format!("{out}/boring_hip_attrs.rs"), "const BORING_HIP_ATTR_WARP_SIZE: c_int = -1;\nconst BORING_HIP_ATTR_MAX_THREADS_PER_BLOCK: c_int = -1;\nconst BORING_HIP_ATTR_SHARED_MEM_PER_BLOCK: c_int = -1;\n").unwrap();
    std::fs::write(format!("{out}/k.co"), "").unwrap();
    println!("cargo:rustc-env=BORING_HIP_CO_PATH={out}/k.co");
}
"#).unwrap();
    }
    let shared = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("screen_accessibility_{target}_target"));
    let fake_nvcc = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/fake-nvcc");
    let path = format!("{}:{}", fake_nvcc.display(), std::env::var("PATH").unwrap_or_default());
    let out = common::SharedCargo::new("screen_accessibility")
        .args(["check", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(shared)
        .env("CARGO_TERM_COLOR", "never")
        .env("PATH", path)
        .output()
        .unwrap();
    assert!(out.status.success(), "[{target}] generated crate does not compile:\n{}", String::from_utf8_lossy(&out.stderr));
}


#[test]
fn wgpu_bridges_the_tree_and_the_requests_to_accesskit() {
    let (project, rs) = build("wgpu", "tree", PROGRAM);
    // AccessKit is wired in: user events wake the loop, the adapter exists before the window is shown.
    assert!(rs.contains("impl ApplicationHandler<__AxEvent> for __App"), "{rs}");
    assert!(rs.contains("EventLoop::<__AxEvent>::with_user_event().build()"), "{rs}");
    assert!(rs.contains(".with_visible(false)"), "{rs}");
    assert!(rs.contains("accesskit_winit::Adapter::with_event_loop_proxy(event_loop, &window, p)"), "{rs}");
    assert!(rs.contains("ax.process_event(w, &event);"), "{rs}");
    assert!(rs.contains("fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: __AxEvent)"), "{rs}");
    assert!(rs.contains("self.__ax_active = true;"), "{rs}");
    assert!(rs.contains("self.__ax_active = false;"), "{rs}");
    // The tree expression is evaluated only while an assistive technology is connected.
    assert!(rs.contains("if self.__ax_active { let __ax_in = threeNodes(self.__frame); self.__ax_push("), "{rs}");
    assert!(rs.contains("update_if_active(|| tu)"), "{rs}");
    // Requests, in arrival order, handed to the program; cleared at the end of the frame.
    assert!(rs.contains("for r in std::mem::take(&mut self.__ax_requests) {"), "{rs}");
    assert!(rs.contains("note(r.id, r.action, r.arg);"), "{rs}");
    assert!(rs.contains("self.__ax_requests.clear();"), "{rs}");
    assert!(rs.contains("fn __boring_ax_request(r: &accesskit::ActionRequest) -> Option<__AccessRequest>"), "{rs}");
    assert!(rs.contains("Action::Click => (\"activate\", String::new())"), "{rs}");
    let cargo = fs::read_to_string(project.join("Cargo.toml")).unwrap();
    assert!(cargo.contains("accesskit = \"0.24\"") && cargo.contains("accesskit_winit = \"0.32\""), "{cargo}");
    cargo_check(&project, "wgpu");
}

#[test]
fn wgpu_without_the_api_has_no_accesskit() {
    let src = PROGRAM.replace("        screen.setAccessibility(threeNodes(screen.frame))\n", "")
        .replace("        for r in screen.accessibilityRequests:\n            note(r.id, r.action, r.arg)\n", "");
    let (project, rs) = build("wgpu", "plain", &src);
    assert!(!rs.contains("accesskit"), "{rs}");
    assert!(!rs.contains("__AxEvent"), "{rs}");
    let cargo = fs::read_to_string(project.join("Cargo.toml")).unwrap();
    assert!(!cargo.contains("accesskit"), "{cargo}");
}

#[test]
fn every_other_target_ignores_the_api() {
    for target in ["simd", "metal", "cuda", "rocm"] {
        let (project, rs) = build(target, "tree", PROGRAM);
        assert!(!rs.contains("setAccessibility") && !rs.contains("set_accessibility") && !rs.contains("accessibilityRequests"), "[{target}] {rs}");
        assert!(!rs.contains("accesskit"), "[{target}] {rs}");
        if target != "metal" || cfg!(target_os = "macos") {
            cargo_check(&project, target);
        }
    }
}
