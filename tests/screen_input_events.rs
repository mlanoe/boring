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

/// System clipboard: read lazily, write on demand.
const CLIPBOARD_LOOP: &str = r#"
kernel:
    loop:
        if screen.key_pressed("v") and screen.clipboard != "":
            k.mode = 1
        if screen.key_pressed("c"):
            screen.setClipboard("copied")
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
    assert!(rs.contains("window.set_ime_allowed(cfg!(not(any(target_os = \"ios\", target_os = \"android\"))));"), "IME must be enabled on desktops:\n{rs}");
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

/// A presented `'unified` field re-uploaded at another size (a resize): the blit must be re-pointed at the
/// reallocated buffer, exactly as for `'surface` (it used to keep showing the old buffer plus blank).
#[test]
fn wgpu_reuploaded_unified_present_buffer_repoints_blit() {
    let src = r#"
let screen = Screen(Dimension(16, 16), title = "unified")

kernel Fill:
    mut [uint]'unified out
    let int width
    let int height

    init(int w, int h):
        out = [0xFF101010 for ..<w * h]
        width = w
        height = h

    def ():
        let col = .block.x * .block_dim.x + .thread.x
        let row = .block.y * .block_dim.y + .thread.y
        if col < width and row < height:
            out[row * width + col] = 0xFF203040

var mut k = Fill(16, 16)

def [uint] blank(int n):
    [0xFF000000 for ..<n]

kernel:
    loop:
        if screen.resized:
            k.out = blank(screen.width * screen.height)
            k.width = screen.width
            k.height = screen.height
        k(block = (16, 16))
        screen.present(k.out)
"#;
    let (project, rs) = build("wgpu", "unified_reupload", src);
    let upload = rs.find("self.k.copy_out_to_device(").expect(&rs);
    let tail = &rs[upload..];
    let end = tail.find('\n').unwrap_or(tail.len());
    assert!(tail[..end].contains("self.__reblit();"), "re-upload of the presented buffer must reblit:\n{}", &tail[..end]);
    assert!(rs.contains("self.k.out_buf.as_entire_binding()"), "{rs}");
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

#[test]
fn wgpu_clipboard_read_write_and_dependency() {
    let (project, rs) = build("wgpu", "clipboard", &events(CLIPBOARD_LOOP));
    assert!(rs.contains("std::sync::Arc::<str>::from(__boring_clipboard_get())"), "{rs}");
    assert!(rs.contains("__boring_clipboard_set(&*("), "{rs}");
    assert!(rs.contains("fn __boring_clipboard_get() -> String"), "{rs}");
    let cargo = fs::read_to_string(project.join("Cargo.toml")).unwrap();
    assert!(cargo.contains("arboard"), "{cargo}");
    cargo_check(&project, "wgpu");
}

#[test]
fn wgpu_without_clipboard_has_no_arboard() {
    let (project, rs) = build("wgpu", "noclip", &events(TEXT_LOOP));
    assert!(!rs.contains("arboard"), "{rs}");
    assert!(!fs::read_to_string(project.join("Cargo.toml")).unwrap().contains("arboard"));
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

/// `examples/game_of_life.br`: a `let [uint]'surface` field (`Render.cells`) must be built
/// `Arc`-wrapped and accept the `render.cells = step.cells_in` kernel-to-kernel buffer copy.
#[test]
fn cuda_rocm_game_of_life_let_surface_field_is_arc_wrapped() {
    let src = fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/game_of_life.br")).unwrap();
    for target in ["cuda", "rocm"] {
        let (project, _) = build(target, "game_of_life", &src);
        cargo_check(&project, target);
    }
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

/// A phone's window is the whole screen and its soft keyboard follows `screen.setImeAllowed` (wgpu): iOS/Android get
/// no requested window size and start with input methods off, desktops keep both.
#[test]
fn wgpu_mobile_window_fills_the_screen_and_keyboard_follows_setimeallowed() {
    let src = events(r#"
kernel:
    loop:
        screen.setImeAllowed(screen.mouse.left)
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#);
    let (project, rs) = build("wgpu", "mobile_window", &src);
    assert!(rs.contains("#[cfg(not(any(target_os = \"ios\", target_os = \"android\")))]\n        let __attrs = __attrs.with_inner_size("), "{rs}");
    assert!(rs.contains("window.set_ime_allowed(cfg!(not(any(target_os = \"ios\", target_os = \"android\"))));"), "{rs}");
    assert!(rs.contains("__ime_on: cfg!(not(any(target_os = \"ios\", target_os = \"android\")))"), "{rs}");
    assert!(rs.contains("if on != self.__ime_on { self.__ime_on = on; w.set_ime_allowed(on); }"), "{rs}");
    cargo_check(&project, "wgpu");
}

/// On iOS the generated host adopts the UIScene life cycle (winit 0.30 does not): a scene delegate class is registered before the
/// event loop, winit's window is attached to the scene UIKit connects, and `objc2` is an iOS-only dependency placed before
/// `[dependencies]` (the project's own dependencies are appended to the end of the manifest).
#[test]
fn wgpu_ios_scene_delegate() {
    let (project, rs) = build("wgpu", "ios_scene", &events(TOUCH_LOOP));
    assert!(rs.contains("mod __boring_scene {"), "{rs}");
    assert!(rs.contains("ClassBuilder::new(c\"BoringSceneDelegate\", NSObject::class())"), "{rs}");
    assert!(rs.contains("sel!(scene:willConnectToSession:options:)"), "{rs}");
    assert!(rs.contains("c\"UISceneDelegate\""), "UIKit refuses a delegate that does not declare the protocols:\n{rs}");
    assert!(rs.contains("#[cfg(target_os = \"ios\")]\n    __boring_scene::register();"), "registered before the event loop:\n{rs}");
    assert!(rs.contains("#[cfg(target_os = \"ios\")]\n        __boring_scene::window_created(&window);"), "{rs}");
    let manifest = fs::read_to_string(project.join("Cargo.toml")).unwrap();
    let ios = manifest.find("[target.'cfg(target_os = \"ios\")'.dependencies]").expect("the iOS dependency table");
    let deps = manifest.find("\n[dependencies]\n").expect("[dependencies]");
    assert!(ios < deps, "[dependencies] must stay the last section:\n{manifest}");
    assert!(manifest[ios..deps].contains("objc2 = \"0.6\""), "{manifest}");
    cargo_check(&project, "wgpu");
}

/// The safe area's insets are readable members of `screen` (wgpu): read from winit's inner rectangle on iOS, 0 elsewhere.
#[test]
fn wgpu_safe_area_members() {
    let src = events(r#"
kernel:
    loop:
        k.tick = screen.safeTop + screen.safeLeft + screen.safeBottom + screen.safeRight
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#);
    let (project, rs) = build("wgpu", "safe_area", &src);
    for (i, member) in ["safeTop", "safeLeft", "safeBottom", "safeRight"].iter().enumerate() {
        assert!(rs.contains(&format!("self.__safe[{i}]")), "screen.{member} -> self.__safe[{i}]:\n{rs}");
    }
    assert!(rs.contains("fn __update_safe(&mut self) {"), "{rs}");
    assert!(rs.contains("w.inner_position()"), "the insets come from winit's inner rectangle:\n{rs}");
    assert!(rs.contains("self.__update_safe();"), "refreshed every frame:\n{rs}");
    cargo_check(&project, "wgpu");
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
    assert!(rs.contains("struct __ScreenSize { width: isize, height: isize, scale: f32 }"), "{rs}");
    assert!(rs.contains("fn __boring_screen_rebuild_k(screen: &__ScreenSize, pal: &Vec<usize>) -> Result<Tiles,"), "{rs}");
    assert!(rs.contains("self.k = __boring_screen_rebuild_k(&__ScreenSize { width: self.__win_w, height: self.__win_h, scale: self.__scale }, &self.pal).expect(\"kernel rebuild failed\"); self.__reblit();"), "{rs}");
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

// ── clipboard on the other hosts ─────────────────────────────────────────────────────────────

fn assert_clipboard_host(target: &str, project: &std::path::Path, rs: &str, set: &str) {
    assert!(rs.contains(set), "[{target}] {rs}");
    assert!(
        rs.contains("fn __boring_clipboard_get() -> String"),
        "[{target}] {rs}"
    );
    assert!(rs.contains("__boring_clipboard_get()"), "[{target}] {rs}");
    let cargo = fs::read_to_string(project.join("Cargo.toml")).unwrap();
    assert!(cargo.contains("arboard = \"3\""), "[{target}] {cargo}");
}

fn assert_no_clipboard(target: &str, project: &std::path::Path, rs: &str) {
    assert!(!rs.contains("arboard"), "[{target}] {rs}");
    assert!(!fs::read_to_string(project.join("Cargo.toml"))
        .unwrap()
        .contains("arboard"));
}

#[test]
fn metal_clipboard_read_write_and_dependency() {
    let (project, rs) = build("metal", "clipboard", &events(CLIPBOARD_LOOP));
    assert_clipboard_host(
        "metal",
        &project,
        &rs,
        "__boring_clipboard_set(&*(\"copied\"));",
    );
    if cfg!(target_os = "macos") {
        cargo_check(&project, "metal");
    }
    let (project, rs) = build("metal", "noclip", &events(TEXT_LOOP));
    assert_no_clipboard("metal", &project, &rs);
}

#[test]
fn cuda_clipboard_read_write_and_dependency() {
    let (project, rs) = build("cuda", "clipboard", &events(CLIPBOARD_LOOP));
    assert_clipboard_host(
        "cuda",
        &project,
        &rs,
        "__boring_clipboard_set(&*(\"copied\"));",
    );
    cargo_check(&project, "cuda");
    let (project, rs) = build("cuda", "noclip", &events(TEXT_LOOP));
    assert_no_clipboard("cuda", &project, &rs);
}

#[test]
fn rocm_clipboard_read_write_and_dependency() {
    let (project, rs) = build("rocm", "clipboard", &events(CLIPBOARD_LOOP));
    assert_clipboard_host(
        "rocm",
        &project,
        &rs,
        "__boring_clipboard_set(&*(\"copied\"));",
    );
    cargo_check(&project, "rocm");
    let (project, rs) = build("rocm", "noclip", &events(TEXT_LOOP));
    assert_no_clipboard("rocm", &project, &rs);
}

#[test]
fn simd_clipboard_read_write_and_dependency() {
    let (project, rs) = build("simd", "clipboard", &events(CLIPBOARD_LOOP));
    assert!(rs.contains("screen.clipboard()"), "{rs}");
    assert!(rs.contains("screen.set_clipboard(&*("), "{rs}");
    assert!(rs.contains("fn __boring_clipboard_get() -> String"), "{rs}");
    assert!(fs::read_to_string(project.join("Cargo.toml"))
        .unwrap()
        .contains("arboard = \"3\""));
    cargo_check(&project, "simd");
    let (project, rs) = build("simd", "noclip", &events(TEXT_LOOP));
    assert!(!rs.contains("arboard"), "{rs}");
    assert!(!fs::read_to_string(project.join("Cargo.toml"))
        .unwrap()
        .contains("arboard"));
}

/// `screen.clipboard` into a `string` function parameter, and a `string` value into `setClipboard`.
const CLIPBOARD_STRINGS_LOOP: &str = r#"
int shout(string s):
    s.length

kernel:
    loop:
        if screen.key_pressed("v"):
            k.mode = shout(screen.clipboard)
        screen.setClipboard(screen.text)
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;

#[test]
fn clipboard_string_conversions_on_other_hosts() {
    for target in ["simd", "metal", "cuda", "rocm"] {
        let (project, _rs) = build(target, "clipstr", &events(CLIPBOARD_STRINGS_LOOP));
        if target != "metal" || cfg!(target_os = "macos") {
            cargo_check(&project, target);
        }
    }
}

// ── touch ────────────────────────────────────────────────────────────────────────────────────
// No touch screen is needed: generation is checked per host, and the shared state machine is
// exercised by injecting synthetic `touch_event`s into the runtime sliced out of a generated crate.

const TOUCH_LOOP: &str = r#"
kernel:
    loop:
        if screen.touching and screen.touchCount >= 2:
            k.mode = 1
        k.tick = screen.touch.x + screen.touch.y
        if screen.touchStarted or screen.touchEnded or screen.touchCancelled:
            k.mode = 2
        if screen.touchCount > 1:
            k.tick = screen.touchAt(1).x - screen.touchAt(1).y
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;

#[test]
fn wgpu_touch_members() {
    let (project, rs) = build("wgpu", "touch", &events(TOUCH_LOOP));
    assert!(rs.contains("WindowEvent::Touch(__boring_t) =>"), "{rs}");
    assert!(rs.contains("self.__input.touch_event(__boring_ph, __boring_t.id, (__boring_t.location.x * self.__buf_w / (ws.width.max(1) as f64)).floor() as isize,"), "buffer-scaled like the cursor:\n{rs}");
    // The scale divisor is the size the surface was last sized to, not `inner_size()` (the safe area on iOS, smaller than
    // the view the touch is located in).
    assert!(rs.contains("let ws = winit::dpi::PhysicalSize::new(self.__win_w.max(1) as u32, self.__win_h.max(1) as u32);"), "{rs}");
    assert!(!rs.contains("let ws = w.inner_size();"), "touches and the cursor must not be scaled by inner_size():\n{rs}");
    assert!(rs.contains("(!self.__input.touches.is_empty())"), "{rs}");
    assert!(rs.contains("(self.__input.touches.len() as isize)"), "{rs}");
    assert!(rs.contains("self.__input.touch_started"), "{rs}");
    assert!(rs.contains("self.__input.touch_x"), "{rs}");
    assert!(rs.contains("self.__input.touch_at_x(1)"), "{rs}");
    cargo_check(&project, "wgpu");
}

#[test]
fn touch_invalidates_on_demand_windows() {
    let src = events(TOUCH_LOOP).replace(r#"title = "input")"#, r#"title = "input", redraw = "demand")"#);
    let (_, rs) = build("wgpu", "touch_demand", &src);
    assert!(rs.contains("WindowEvent::Focused(_) | WindowEvent::Touch(_)) {"), "{rs}");
    for target in ["metal", "cuda", "rocm"] {
        let (_, rs) = build(target, "touch_demand", &src);
        assert!(rs.contains("winit::event::WindowEvent::Focused(_) | winit::event::WindowEvent::Touch(_), .. }) { boring_dirty = true; }"), "[{target}] {rs}");
    }
}

#[test]
fn simd_touch_members() {
    let (project, rs) = build("simd", "touch", &events(TOUCH_LOOP));
    assert!(rs.contains("WindowEvent::Touch(t)"), "{rs}");
    assert!(rs.contains("touch_event(phase, t.id,"), "{rs}");
    assert!(rs.contains("fn touch_at(&self, i: isize) -> Touch"), "{rs}");
    assert!(rs.contains("screen.touching"), "{rs}");
    assert!(rs.contains("screen.touchCount"), "{rs}");
    assert!(rs.contains("screen.touch.x"), "{rs}");
    assert!(rs.contains("screen.touch_at(1).x"), "{rs}");
    cargo_check(&project, "simd");
}

fn assert_winit028_touch(target: &str, rs: &str) {
    assert!(rs.contains("winit::event::WindowEvent::Touch(__boring_t)"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.touch_event(__boring_ph, __boring_t.id, (__boring_t.location.x * boring_buf_w / (__boring_ws.width.max(1) as f64)).floor() as isize,"), "[{target}] {rs}");
    assert!(rs.contains("(!boring_input.touches.is_empty())"), "[{target}] {rs}");
    assert!(rs.contains("(boring_input.touches.len() as isize)"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.touch_started"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.touch_ended"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.touch_cancelled"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.touch_x"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.touch_y"), "[{target}] {rs}");
    assert!(rs.contains("boring_input.touch_at_x(1)"), "[{target}] {rs}");
}

#[test]
fn metal_touch_members() {
    let (project, rs) = build("metal", "touch", &events(TOUCH_LOOP));
    assert_winit028_touch("metal", &rs);
    if cfg!(target_os = "macos") {
        cargo_check(&project, "metal");
    }
}

#[test]
fn cuda_touch_members() {
    let (project, rs) = build("cuda", "touch", &events(TOUCH_LOOP));
    assert_winit028_touch("cuda", &rs);
    cargo_check(&project, "cuda");
}

#[test]
fn rocm_touch_members() {
    let (project, rs) = build("rocm", "touch", &events(TOUCH_LOOP));
    assert_winit028_touch("rocm", &rs);
    cargo_check(&project, "rocm");
}

#[test]
fn touch_state_machine_with_synthetic_events() {
    let (_, generated) = build("simd", "touch_source", &events(TOUCH_LOOP));
    let from = generated.find("// ---- Screen input (shared by every Screen host) ----").expect("input runtime");
    let to = generated.find("fn __boring_vk_name").expect("key table");
    let runtime = generated[from..to].trim_end_matches("#[allow(dead_code)]\n").to_string();
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_input_events").join("touch_state");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let src = format!(
        "{runtime}\nfn main() {{\n    \
         let mut i = __BoringInput::new();\n    \
         // Tap shorter than a frame: started and ended in the same frame, nothing left down.\n    \
         i.touch_event(0, 7, 10, 20); i.touch_event(2, 7, 11, 21);\n    \
         let s = i.snapshot();\n    \
         assert!(s.touch_started && s.touch_ended && !s.touch_cancelled && s.touches.is_empty());\n    \
         assert_eq!((s.touch_x, s.touch_y), (11, 21));\n    \
         assert!(!i.touch_started && !i.touch_ended);\n    \
         // Two fingers: the first placed is primary while it stays down.\n    \
         i.touch_event(0, 1, 5, 6); i.touch_event(0, 2, 50, 60); i.touch_event(1, 2, 51, 61);\n    \
         let s = i.snapshot();\n    \
         assert!(s.touch_started && s.touches.len() == 2 && (s.touch_x, s.touch_y) == (5, 6));\n    \
         assert_eq!((s.touch_at_x(1), s.touch_at_y(1), s.touch_at_x(9)), (51, 61, 0));\n    \
         // Secondary finger lifting is not the primary ending.\n    \
         i.touch_event(2, 2, 52, 62);\n    \
         let s = i.snapshot();\n    \
         assert!(!s.touch_ended && s.touches.len() == 1);\n    \
         // The primary moves; a new finger placed meanwhile does not become primary.\n    \
         i.touch_event(1, 1, 8, 9); i.touch_event(0, 3, 100, 100); i.touch_event(1, 3, 101, 101);\n    \
         let s = i.snapshot();\n    \
         assert!(!s.touch_started && (s.touch_x, s.touch_y) == (8, 9) && s.touches.len() == 2);\n    \
         // Primary cancelled: cancelled, not ended; last position kept; finger 3 still down.\n    \
         i.touch_event(3, 1, 9, 10);\n    \
         let s = i.snapshot();\n    \
         assert!(s.touch_cancelled && !s.touch_ended && s.touches.len() == 1 && (s.touch_x, s.touch_y) == (9, 10));\n    \
         // Unknown ids never panic or change state.\n    \
         i.touch_event(1, 99, 1, 1); i.touch_event(2, 99, 1, 1);\n    \
         let s = i.snapshot();\n    \
         assert!(!s.touch_ended && s.touches.len() == 1);\n}}\n"
    );
    let rs = tmp.join("main.rs");
    fs::write(&rs, src).unwrap();
    let exe = tmp.join("touch_state");
    let out = Command::new("rustc").args(["--edition", "2021", "-A", "warnings", "-o"]).arg(&exe).arg(&rs).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let run = Command::new(&exe).output().unwrap();
    assert!(run.status.success(), "{}{}", String::from_utf8_lossy(&run.stdout), String::from_utf8_lossy(&run.stderr));
}
