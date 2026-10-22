// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `Screen` input beyond keys and the mouse position (docs/gpu-display.md, "Input"): typed text,
// IME composition, modifier keys, wheel, button/key edges, key repeat, focus, a live window size
// and kernel reallocation after a resize (`k = K(Dimension(screen.width, screen.height))`),
// on every Screen host (simd, Metal, CUDA, ROCm, wgpu).
//
// wgpu / simd / CUDA (with `tools/fake-nvcc`) generated crates are `cargo check`ed, Metal too on
// macOS; ROCm too, with its `build.rs` (hipcc + device probe) replaced by a stub.
// No window is opened.

mod common;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const KERNEL: &str = r#"
let screen = Screen(Dimension(16, 16), title = "input")

kernel Fill:
    mut [uint]'surface pixels
    let Dimension dim
    var int tick
    var float32 scroll
    var int mode

    init(Dimension d):
        pixels = [0xFF101010 for ..<d.width * d.height]
        dim = d
        tick = 0
        scroll = 0.0
        mode = 0

    def ():
        let col = .block.x * .block_dim.x + .thread.x
        let row = .block.y * .block_dim.y + .thread.y
        if col < dim.width and row < dim.height:
            pixels[row * dim.width + col] = 0xFF000000 | uint(tick + mode)

var mut k = Fill(Dimension(16, 16))
"#;

/// Every input member, a resize-driven reallocation and a non-literal key.
const EVENTS_LOOP: &str = r#"
kernel:
    loop:
        k.tick = screen.mouse.x
        k.scroll = screen.mouse.scroll_y + screen.mouse.scroll_x
        if screen.resized:
            k = Fill(Dimension(screen.width, screen.height))
        if screen.key_repeat("Backspace") or screen.key_released("a") or screen.mouse_pressed("left") or screen.mouse_released("right"):
            k.mode = 1
        if screen.shift and screen.ctrl and screen.alt and screen.meta:
            k.mode = 2
        if screen.focused and (screen.focus_gained or screen.focus_lost):
            k.mode = 3
        if screen.marked_start >= 0 and screen.marked_end > screen.marked_start:
            k.mode = 4
        if screen.key("Left") or screen.key("\x1B[D") or screen.key_pressed("Enter") or screen.key_pressed("\r"):
            k.mode = 5
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;

/// The string-valued members.
const TEXT_LOOP: &str = r#"
kernel:
    loop:
        if screen.text == "x":
            k.mode = 1
        if screen.marked == screen.last_key:
            k.mode = 2
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;

fn build(target: &str, name: &str, src: &str) -> (PathBuf, String) {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_input_events").join(format!("{name}_{target}"));
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

fn cargo_check(project: &PathBuf, target: &str) {
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
    let shared = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("screen_input_events_{target}_target"));
    let fake_nvcc = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/fake-nvcc");
    let path = format!("{}:{}", fake_nvcc.display(), std::env::var("PATH").unwrap_or_default());
    let out = common::SharedCargo::new("screen_input_events")
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

fn events(loop_body: &str) -> String { format!("{KERNEL}{loop_body}") }

// ── wgpu ─────────────────────────────────────────────────────────────────────────────────────

#[test]
fn wgpu_input_events_and_resize_reallocation() {
    let (project, rs) = build("wgpu", "events", &events(EVENTS_LOOP));
    // The shared runtime and the canonical key names.
    assert!(rs.contains("struct __BoringInput"), "{rs}");
    assert!(rs.contains("fn __boring_key_name(key: &str) -> String"), "{rs}");
    assert!(rs.contains("fn __boring_logical_key_name(key: &winit::keyboard::Key) -> String"), "{rs}");
    // Events reach it.
    assert!(rs.contains("window.set_ime_allowed(true);"), "IME must be enabled:\n{rs}");
    assert!(rs.contains("WindowEvent::Ime(ime)"), "{rs}");
    assert!(rs.contains("self.__input.ime_commit(&s)"), "{rs}");
    assert!(rs.contains("WindowEvent::ModifiersChanged(m)"), "{rs}");
    assert!(rs.contains("WindowEvent::MouseWheel { delta, .. }"), "{rs}");
    assert!(rs.contains("WindowEvent::Focused(f)"), "{rs}");
    assert!(rs.contains("self.__input.key_down(&key_str);"), "{rs}");
    assert!(rs.contains("self.__input.key_up(&key_str);"), "{rs}");
    assert!(rs.contains("self.__input.mouse_button(1, down);"), "{rs}");
    assert!(rs.contains("self.__input.end_frame();"), "the per-frame events are cleared after the body:\n{rs}");
    // Members.
    assert!(rs.contains("self.k.tick = (self.__mouse_x) as i32;"), "int field narrowed to its 32-bit mirror:\n{rs}");
    assert!(rs.contains("self.__input.scroll_y"), "{rs}");
    assert!(rs.contains("self.__input.key_repeat(&*(std::sync::Arc::<str>::from(\"Backspace\")))"), "{rs}");
    assert!(rs.contains("self.__input.key_released(&*("), "{rs}");
    assert!(rs.contains("self.__input.mouse_pressed(&*("), "{rs}");
    assert!(rs.contains("self.__input.mouse_released(&*("), "{rs}");
    assert!(rs.contains("(self.__input.mods & 1 != 0)"), "{rs}");
    assert!(rs.contains("(self.__input.mods & 8 != 0)"), "{rs}");
    assert!(rs.contains("self.__input.focus_gained"), "{rs}");
    assert!(rs.contains("self.__input.marked_start"), "{rs}");
    // Live window size; `screen.resized` is an ordinary loop condition (it used to emit a bare `screen`).
    assert!(rs.contains("if self.__resized {"), "{rs}");
    assert!(!rs.contains("screen.resized"), "{rs}");
    assert!(rs.contains("self.__win_w = size.width as isize;"), "{rs}");
    // Reallocation: new kernel, swapchain + blit follow the window, mouse coordinates the buffer.
    assert!(rs.contains("self.k = Fill::new((self.__win_w) as i32, (self.__win_h) as i32,"), "{rs}");
    assert!(rs.contains("self.__reblit();"), "{rs}");
    assert!(rs.contains("fn __reblit(&mut self)"), "{rs}");
    assert!(rs.contains("surface.configure(&self.device, cfg);"), "{rs}");
    assert!(rs.contains("self.__buf_w"), "{rs}");
    // Key queries go through the runtime normalizer.
    assert!(rs.contains("self.__keys.contains(&__boring_key_name(\"Left\"))"), "{rs}");
    assert!(rs.contains("self.__keys_pressed.contains(&__boring_key_name(\"Enter\"))"), "{rs}");
    assert!(rs.contains("self.__keys.contains(&__boring_key_name(\"\\\\x1B\")) { event_loop.exit(); }"), "{rs}");
    cargo_check(&project, "wgpu");
}

#[test]
fn wgpu_text_members_are_strings() {
    let (project, rs) = build("wgpu", "text", &events(TEXT_LOOP));
    assert!(rs.contains("std::sync::Arc::<str>::from(self.__input.text())"), "{rs}");
    assert!(rs.contains("std::sync::Arc::<str>::from(self.__input.marked.clone())"), "{rs}");
    assert!(rs.contains("std::sync::Arc::<str>::from(self.__input.last_key.clone())"), "{rs}");
    cargo_check(&project, "wgpu");
}

// ── simd ─────────────────────────────────────────────────────────────────────────────────────

#[test]
fn simd_input_events_and_resize_reallocation() {
    let (project, rs) = build("simd", "events", &events(EVENTS_LOOP));
    assert!(rs.contains("struct __BoringInput"), "{rs}");
    assert!(rs.contains("fn __boring_vk_name(key: winit::event::VirtualKeyCode) -> String"), "{rs}");
    assert!(rs.contains("built.set_ime_allowed(true);"), "{rs}");
    assert!(rs.contains("WindowEvent::ReceivedCharacter(c)"), "{rs}");
    assert!(rs.contains("WindowEvent::Ime(ime)"), "{rs}");
    assert!(rs.contains("WindowEvent::ModifiersChanged(m)"), "{rs}");
    assert!(rs.contains("WindowEvent::MouseWheel { delta, .. }"), "{rs}");
    assert!(rs.contains("WindowEvent::Focused(focused)"), "{rs}");
    assert!(rs.contains("fn key_repeat<T: AsRef<str>>(&self, key: T) -> bool"), "{rs}");
    assert!(rs.contains("fn mouse_pressed<T: AsRef<str>>(&self, button: T) -> bool"), "{rs}");
    assert!(rs.contains("screen.key_repeat(\"Backspace\")"), "{rs}");
    assert!(rs.contains("screen.mouse.scroll_y"), "{rs}");
    assert!(rs.contains("screen.resized"), "{rs}");
    // The program reallocates its kernel for the window's size; the blit follows the frame's size.
    assert!(rs.contains("k = Fill::new((screen.width) as i32, (screen.height) as i32,"), "{rs}");
    assert!(rs.contains("frame_dims"), "{rs}");
    cargo_check(&project, "simd");
}

#[test]
fn simd_text_members_are_strings() {
    let (project, rs) = build("simd", "text", &events(TEXT_LOOP));
    assert!(rs.contains("text: std::sync::Arc<str>"), "{rs}");
    assert!(rs.contains("self.text = self.input.text().into();"), "{rs}");
    cargo_check(&project, "simd");
}

// ── Metal / CUDA / ROCm (winit 0.28 hosts) ───────────────────────────────────────────────────

fn assert_winit028_host(target: &str, rs: &str) {
    assert!(rs.contains("struct __BoringInput"), "[{target}] {rs}");
    assert!(rs.contains("fn __boring_vk_name(key: winit::event::VirtualKeyCode) -> String"), "[{target}] {rs}");
    assert!(rs.contains("fn boring_key_str(key: winit::event::VirtualKeyCode) -> String { __boring_vk_name(key) }"), "[{target}] {rs}");
    assert!(rs.contains("boring_window.set_ime_allowed(true);"), "[{target}] {rs}");
    assert!(rs.contains("let mut boring_input = __BoringInput::new();"), "[{target}] {rs}");
    assert!(rs.contains("winit::event::WindowEvent::ReceivedCharacter(__boring_c)"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.typed(__boring_c);"), "[{target}] {rs}");
    assert!(rs.contains("winit::event::WindowEvent::Ime(__boring_ime)"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.ime_commit(&__boring_s)"), "[{target}] {rs}");
    assert!(rs.contains("winit::event::WindowEvent::ModifiersChanged(__boring_m)"), "[{target}] {rs}");
    assert!(rs.contains("winit::event::WindowEvent::MouseWheel { delta: __boring_d, .. }"), "[{target}] {rs}");
    assert!(rs.contains("winit::event::WindowEvent::Focused(__boring_f)"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.key_down(&__boring_ks);"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.key_up(&__boring_ks);"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.mouse_button(match __boring_mb {"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.end_frame();"), "[{target}] {rs}");
    // Members.
    assert!(rs.contains("boring_input.key_repeat(&*(\"Backspace\"))"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.key_released(&*("), "[{target}] {rs}");
    assert!(rs.contains("boring_input.mouse_pressed(&*("), "[{target}] {rs}");
    assert!(rs.contains("boring_input.mouse_released(&*("), "[{target}] {rs}");
    assert!(rs.contains("boring_input.scroll_y"), "[{target}] {rs}");
    assert!(rs.contains("(boring_input.mods & 1 != 0)"), "[{target}] {rs}");
    assert!(rs.contains("(boring_input.mods & 8 != 0)"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.focus_lost"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.marked_end"), "[{target}] {rs}");
    // Keys are queried by canonical name, whatever the spelling.
    assert!(rs.contains("boring_keys.contains(&__boring_key_name(&*(\"Left\")))"), "[{target}] {rs}");
    assert!(rs.contains("boring_keys_pressed.contains(&__boring_key_name(&*(\"Enter\")))"), "[{target}] {rs}");
    // Resize: live size, kernel reallocated for it, mouse follows the buffer.
    assert!(rs.contains("if boring_screen_resized {"), "[{target}] {rs}");
    assert!(rs.contains("Dimension((boring_screen_width) as u32, (boring_screen_height) as u32)"), "[{target}] {rs}");
    assert!(rs.contains(".expect(\"kernel construction failed\")"), "[{target}] {rs}");
    assert!(rs.contains("boring_buf_w = ("), "[{target}] {rs}");
}

#[test]
fn metal_input_events_and_resize_reallocation() {
    let (project, rs) = build("metal", "events", &events(EVENTS_LOOP));
    assert_winit028_host("metal", &rs);
    assert!(rs.contains("k = Fill::new(boring_device.clone(), Dimension("), "{rs}");
    assert!(rs.contains("layer.set_drawable_size("), "the drawable follows the presented buffer:\n{rs}");
    if cfg!(target_os = "macos") {
        cargo_check(&project, "metal");
    }
}

#[test]
fn metal_text_members_are_strings() {
    let (project, rs) = build("metal", "text", &events(TEXT_LOOP));
    assert!(rs.contains("boring_input.text()"), "{rs}");
    if cfg!(target_os = "macos") {
        cargo_check(&project, "metal");
    }
}

#[test]
fn cuda_input_events_and_resize_reallocation() {
    let (project, rs) = build("cuda", "events", &events(EVENTS_LOOP));
    assert_winit028_host("cuda", &rs);
    assert!(rs.contains("k = Some(Fill::new(boring_gpu_ctx(), Dimension("), "{rs}");
    assert!(rs.contains("surface.resize(w, h)"), "the softbuffer surface follows the presented buffer:\n{rs}");
    cargo_check(&project, "cuda");
}

#[test]
fn cuda_text_members_are_strings() {
    let (project, rs) = build("cuda", "text", &events(TEXT_LOOP));
    assert!(rs.contains("boring_input.text()"), "{rs}");
    cargo_check(&project, "cuda");
}

#[test]
fn rocm_input_events_and_resize_reallocation() {
    let (project, rs) = build("rocm", "events", &events(EVENTS_LOOP));
    assert_winit028_host("rocm", &rs);
    assert!(rs.contains("k = Some(Fill::new(boring_gpu_ctx(), Dimension("), "{rs}");
    assert!(rs.contains("surface.resize(w, h)"), "the softbuffer surface follows the presented buffer:\n{rs}");
    cargo_check(&project, "rocm");
}

#[test]
fn rocm_text_members_and_event_loops() {
    let (project, _) = build("rocm", "text", &events(TEXT_LOOP));
    cargo_check(&project, "rocm");
    let (project, _) = build("rocm", "queue", &queue_program());
    cargo_check(&project, "rocm");
}

// ── Key events, IME area, string hand-off, setup-kernel rebuild ──────────────────────────────

/// A `string` function called from the loop with `screen.text`, a literal and the loop variable of
/// `for k in screen.key_events:`; the IME area hint.
fn queue_program() -> String {
    format!("{KERNEL}{}", r#"
def note(string t, int n):
    print "{t} {n}"

kernel:
    loop:
        note(screen.text, screen.frame)
        note("literal", 1)
        for key in screen.key_events:
            note(key, 2)
            if key == "Tab":
                k.mode = 1
        for up in screen.key_up_events:
            note(up, 3)
        screen.setImeArea(screen.mouse.x, screen.mouse.y, 4, 12)
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#)
}

#[test]
fn every_target_iterates_key_events_and_hands_strings_to_functions() {
    for target in ["wgpu", "simd", "metal", "cuda", "rocm"] {
        let (project, rs) = build(target, "queue", &queue_program());
        match target {
            "wgpu" => {
                assert!(rs.contains("for __ev in self.__input.key_repeat.clone() {"), "{rs}");
                assert!(rs.contains("let key = std::sync::Arc::<str>::from(__ev.as_str());"), "{rs}");
                assert!(rs.contains("note(key.clone(), 2)"), "the loop variable is passed by clone:\n{rs}");
                assert!(rs.contains("set_ime_cursor_area("), "{rs}");
            }
            "simd" => {
                assert!(rs.contains("for key in screen.key_events.iter().cloned()"), "{rs}");
                assert!(rs.contains("fn set_ime_area(&self"), "{rs}");
                assert!(rs.contains("screen.set_ime_area("), "{rs}");
            }
            _ => {
                assert!(rs.contains("for key in boring_input.key_repeat.clone() {"), "[{target}] {rs}");
                assert!(rs.contains("for up in boring_input.key_released.clone() {"), "[{target}] {rs}");
                // This host's literals and `screen.text` are `&str`/`String`: converted to the callee's `string`.
                assert!(rs.contains("note(std::sync::Arc::<str>::from(boring_input.text().as_str()),"), "[{target}] {rs}");
                assert!(rs.contains("note(std::sync::Arc::<str>::from(\"literal\"),"), "[{target}] {rs}");
                assert!(rs.contains("note(std::sync::Arc::<str>::from(key.as_str()),"), "[{target}] {rs}");
                assert!(rs.contains("boring_window.set_ime_position("), "[{target}] {rs}");
            }
        }
        if target != "metal" || cfg!(target_os = "macos") {
            cargo_check(&project, target);
        }
    }
}

/// A kernel built by the program's setup code from real constructor arguments is rebuilt, for the
/// new window size, by a generated function (wgpu).
#[test]
fn wgpu_reassigning_a_setup_built_kernel_rebuilds_it() {
    let src = r#"
let width = 16
let height = 16

def [uint] palette():
    [0xFF0000FF, 0xFF00FF00]

let screen = Screen(Dimension(width, height), title = "rebuild")
let pal = palette()

kernel Tiles:
    mut [uint]'surface out
    let [uint]'global colors
    var int w
    var int h

    init([uint] c, int ww, int hh):
        out = [0 for ..<ww * hh]
        colors = c
        w = ww
        h = hh

    def ():
        let col = .block.x * .block_dim.x + .thread.x
        let row = .block.y * .block_dim.y + .thread.y
        if col < w and row < h:
            out[row * w + col] = colors[col % 2]

mut k = Tiles(pal, width, height)

kernel:
    loop:
        if screen.resized:
            k = Tiles(pal, screen.width, screen.height)
        k(block = (4, 4))
        screen.present(k.out)
        if screen.key("\x1B"):
            break
"#;
    let (project, rs) = build("wgpu", "rebuild", src);
    assert!(rs.contains("struct __ScreenSize { width: isize, height: isize }"), "{rs}");
    assert!(rs.contains("fn __boring_screen_rebuild_k(screen: &__ScreenSize, pal: &Vec<usize>) -> Result<Tiles,"), "{rs}");
    assert!(rs.contains("self.k = __boring_screen_rebuild_k(&__ScreenSize { width: self.__win_w, height: self.__win_h }, &self.pal).expect(\"kernel rebuild failed\"); self.__reblit();"), "{rs}");
    cargo_check(&project, "wgpu");
}

#[test]
fn wgpu_surface_field_reupload_follows_the_window() {
    let src = format!("{KERNEL}{}", r#"
def [uint] blank(int n):
    [0xFF000000 for ..<n]

kernel:
    loop:
        if screen.resized:
            k.dim = Dimension(screen.width, screen.height)
            k.pixels = blank(screen.width * screen.height)
        k(block = (4, 4))
        screen.present(k.pixels)
"#);
    // `k.dim = ...` is a `Dimension` field assignment: not supported on the host; the field form of
    // the re-upload is what this test pins, so keep to the array.
    let src = src.replace("            k.dim = Dimension(screen.width, screen.height)
", "");
    let (project, rs) = build("wgpu", "reupload", &src);
    assert!(rs.contains("self.k.copy_pixels_to_device("), "{rs}");
    assert!(rs.contains("self.__reblit();"), "{rs}");
    cargo_check(&project, "wgpu");
}

// ── Key names ────────────────────────────────────────────────────────────────────────────────

/// The canonical-name normalizer is plain Rust shipped inside every generated crate; run it.
#[test]
fn key_name_normalizer_accepts_every_spelling() {
    // The shared runtime, sliced out of a generated simd crate (it ends where the winit 0.28 key table starts).
    let (_, generated) = build("simd", "key_names_source", &events(TEXT_LOOP));
    let from = generated.find("// ---- Screen input (shared by every Screen host) ----").expect("input runtime");
    let to = generated.find("fn __boring_vk_name").expect("key table");
    let runtime = || generated[from..to].trim_end_matches("#[allow(dead_code)]\n").to_string();
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_input_events").join("key_names");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let src = format!(
        "{}\nfn main() {{\n    let n = |k: &str| __boring_key_name(k);\n    \
         assert_eq!(n(\"\\x1B\"), \"Escape\"); assert_eq!(n(\"\\\\x1B\"), \"Escape\"); assert_eq!(n(\"Esc\"), \"Escape\");\n    \
         assert_eq!(n(\" \"), \" \"); assert_eq!(n(\"Space\"), \" \");\n    \
         assert_eq!(n(\"\\r\"), \"Enter\"); assert_eq!(n(\"\\\\n\"), \"Enter\"); assert_eq!(n(\"return\"), \"Enter\");\n    \
         assert_eq!(n(\"\\t\"), \"Tab\"); assert_eq!(n(\"\\x08\"), \"Backspace\"); assert_eq!(n(\"\\x7F\"), \"Delete\");\n    \
         assert_eq!(n(\"\\x1B[A\"), \"Up\"); assert_eq!(n(\"ArrowDown\"), \"Down\"); assert_eq!(n(\"\\x1B[D\"), \"Left\"); assert_eq!(n(\"right\"), \"Right\");\n    \
         assert_eq!(n(\"pageup\"), \"PageUp\"); assert_eq!(n(\"f5\"), \"F5\"); assert_eq!(n(\"F12\"), \"F12\");\n    \
         assert_eq!(n(\"A\"), \"a\"); assert_eq!(n(\"7\"), \"7\"); assert_eq!(n(\"Ctrl\"), \"Control\"); assert_eq!(n(\"cmd\"), \"Meta\");\n    \
         // The state: text from key events and from the IME never doubles; edges are per frame.\n    \
         let mut i = __BoringInput::new();\n    \
         i.typed('a'); i.typed('\\r'); i.ime_commit(\"a\");\n    \
         assert_eq!(i.text(), \"a\");\n    \
         i.ime_preedit(\"ni\", Some((0, 2))); assert_eq!((i.marked.as_str(), i.marked_start, i.marked_end), (\"ni\", 0, 2));\n    \
         i.ime_commit(\"你\"); assert_eq!(i.marked, \"\"); assert_eq!(i.marked_start, -1); assert_eq!(i.text(), \"a你\");\n    \
         i.key_down(\"Backspace\"); i.key_down(\"Backspace\"); i.key_up(\"a\");\n    \
         assert!(i.key_repeat(\"backspace\") && i.key_released(\"A\") && !i.key_released(\"b\"));\n    \
         assert_eq!(i.last_key, \"Backspace\");\n    \
         i.mouse_button(1, true); i.mouse_button(2, false); i.scroll(1.0, -2.0); i.set_mods(true, false, false, true); i.set_focus(false);\n    \
         let s = i.snapshot();\n    \
         assert!(s.mouse_pressed(\"left\") && s.mouse_released(\"RIGHT\") && !s.mouse_pressed(\"right\"));\n    \
         assert_eq!((s.scroll_x, s.scroll_y, s.mods, s.focused, s.focus_lost), (1.0, -2.0, 9, false, true));\n    \
         // Snapshot cleared the per-frame part, kept the live part.\n    \
         assert!(!i.mouse_pressed(\"left\") && !i.key_repeat(\"Backspace\") && i.text().is_empty() && !i.focus_lost);\n    \
         assert_eq!((i.mods, i.focused), (9, false));\n}}\n",
        runtime()
    );
    let rs = tmp.join("main.rs");
    fs::write(&rs, src).unwrap();
    let exe = tmp.join("key_names");
    let out = Command::new("rustc").args(["--edition", "2021", "-A", "warnings", "-o"]).arg(&exe).arg(&rs).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let run = Command::new(&exe).output().unwrap();
    assert!(run.status.success(), "{}{}", String::from_utf8_lossy(&run.stdout), String::from_utf8_lossy(&run.stderr));
}
