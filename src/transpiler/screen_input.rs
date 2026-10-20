// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Shared `Screen` input runtime (docs/gpu-display.md, "Input"): one Rust snippet every Screen
// host (simd, Metal, CUDA, ROCm, wgpu) pastes into its generated crate, plus the Boring-source
// -> Rust lowering of the `screen.*` input members. The hosts differ only in how winit events
// reach it (winit 0.28 on simd/Metal/CUDA/ROCm, 0.30 on wgpu) and in where the state lives; the
// state, the key naming and the per-frame semantics are the same everywhere.
//
// Semantics: events accumulate between frames; the frame body reads them; the host clears the
// transient part (`end_frame`) once the body has run. `simd` snapshots at `Screen::present`.

/// The state struct, the canonical key-name normalizer and nothing winit-specific.
pub(super) const INPUT_RUNTIME: &str = r##"
// ---- Screen input (shared by every Screen host) ----
#[allow(dead_code)]
#[derive(Clone, Default)]
struct __BoringInput {
    // Live state.
    mods: u8,            // 1 shift, 2 ctrl, 4 alt, 8 meta (Command / Windows key)
    focused: bool,
    marked: String,      // IME composition (pre-edit) text, "" when none
    marked_start: isize, // caret/selection inside `marked` (byte offsets), -1 when the IME gave none
    marked_end: isize,
    // Per-frame events (cleared by `end_frame`).
    focus_gained: bool,
    focus_lost: bool,
    text_chars: String,  // characters typed (from key events)
    text_commit: String, // text committed by the IME / input method
    last_key: String,
    key_repeat: Vec<String>,   // key went down or auto-repeated, in arrival order
    key_released: Vec<String>,
    btn_pressed: u8,     // 1 left, 2 right, 4 middle
    btn_released: u8,
    scroll_x: f32,       // pixels; a wheel notch is 40
    scroll_y: f32,       // positive = wheel rolled away from the user (content moves down)
}
#[allow(dead_code)]
impl __BoringInput {
    fn new() -> Self { __BoringInput { focused: true, marked_start: -1, marked_end: -1, ..Default::default() } }
    // Text typed this frame. An input method reports the same characters through both
    // channels, so the committed text wins when there is any (never both, never twice).
    fn text(&self) -> String {
        if self.text_commit.is_empty() { self.text_chars.clone() } else { self.text_commit.clone() }
    }
    fn key_down(&mut self, name: &str) { self.last_key = name.to_string(); self.key_repeat.push(name.to_string()); }
    fn key_up(&mut self, name: &str) { self.key_released.push(name.to_string()); }
    fn key_repeat(&self, name: &str) -> bool { let n = __boring_key_name(name); self.key_repeat.iter().any(|k| *k == n) }
    fn key_released(&self, name: &str) -> bool { let n = __boring_key_name(name); self.key_released.iter().any(|k| *k == n) }
    fn set_mods(&mut self, shift: bool, ctrl: bool, alt: bool, meta: bool) {
        self.mods = (shift as u8) | ((ctrl as u8) << 1) | ((alt as u8) << 2) | ((meta as u8) << 3);
    }
    fn set_focus(&mut self, focused: bool) {
        if focused != self.focused { if focused { self.focus_gained = true } else { self.focus_lost = true } }
        self.focused = focused;
    }
    fn typed(&mut self, c: char) { if !c.is_control() { self.text_chars.push(c); } }
    fn ime_preedit(&mut self, text: &str, cursor: Option<(usize, usize)>) {
        self.marked = text.to_string();
        let (s, e) = match cursor { Some((s, e)) if !text.is_empty() => (s as isize, e as isize), _ => (-1, -1) };
        self.marked_start = s;
        self.marked_end = e;
    }
    fn ime_commit(&mut self, text: &str) { self.text_commit.push_str(text); self.ime_preedit("", None); }
    fn mouse_button(&mut self, bit: u8, down: bool) {
        if down { self.btn_pressed |= bit } else { self.btn_released |= bit }
    }
    fn button_bit(name: &str) -> u8 {
        match name.to_ascii_lowercase().as_str() { "left" => 1, "right" => 2, "middle" => 4, _ => 0 }
    }
    fn mouse_pressed(&self, name: &str) -> bool { self.btn_pressed & Self::button_bit(name) != 0 }
    fn mouse_released(&self, name: &str) -> bool { self.btn_released & Self::button_bit(name) != 0 }
    fn scroll(&mut self, dx: f32, dy: f32) { self.scroll_x += dx; self.scroll_y += dy; }
    fn end_frame(&mut self) {
        self.focus_gained = false;
        self.focus_lost = false;
        self.text_chars.clear();
        self.text_commit.clear();
        self.last_key.clear();
        self.key_repeat.clear();
        self.key_released.clear();
        self.btn_pressed = 0;
        self.btn_released = 0;
        self.scroll_x = 0.0;
        self.scroll_y = 0.0;
    }
    // The state as the frame body sees it (live part copied, per-frame part moved out).
    fn snapshot(&mut self) -> __BoringInput { let out = self.clone(); self.end_frame(); out }
}
// Canonical key names, used by every host both when an event arrives and when `screen.key(...)`
// & co. are queried: printable keys are the lowercase character, the rest are
// "Escape" "Enter" "Tab" "Backspace" "Delete" "Insert" "Left" "Right" "Up" "Down" "Home" "End"
// "PageUp" "PageDown" "F1".."F12" "Shift" "Control" "Alt" "Meta" "CapsLock", and " " for space.
// A query also accepts the control-character spellings ("\x1B", "\r", "\n", "\t", "\x08", "\x7F",
// the ANSI arrows "\x1B[A".."\x1B[D"), the winit-style spellings ("ArrowLeft", "Return", "Space",
// "Esc") and any letter case.
#[allow(dead_code)]
fn __boring_key_name(key: &str) -> String {
    let raw = key.replace("\\x1B", "\x1B").replace("\\x1b", "\x1B").replace("\\x08", "\x08")
        .replace("\\x7F", "\x7F").replace("\\x7f", "\x7F").replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t");
    match raw.as_str() {
        "\x1B" => return "Escape".into(),
        " " => return " ".into(),
        "\r" | "\n" => return "Enter".into(),
        "\t" => return "Tab".into(),
        "\x08" => return "Backspace".into(),
        "\x7F" => return "Delete".into(),
        "\x1B[A" => return "Up".into(),
        "\x1B[B" => return "Down".into(),
        "\x1B[C" => return "Right".into(),
        "\x1B[D" => return "Left".into(),
        _ => {}
    }
    let lower = raw.to_ascii_lowercase();
    match lower.as_str() {
        "escape" | "esc" => "Escape".into(),
        "space" | "spacebar" => " ".into(),
        "enter" | "return" => "Enter".into(),
        "tab" => "Tab".into(),
        "backspace" | "back" => "Backspace".into(),
        "delete" | "del" => "Delete".into(),
        "insert" => "Insert".into(),
        "left" | "arrowleft" => "Left".into(),
        "right" | "arrowright" => "Right".into(),
        "up" | "arrowup" => "Up".into(),
        "down" | "arrowdown" => "Down".into(),
        "home" => "Home".into(),
        "end" => "End".into(),
        "pageup" => "PageUp".into(),
        "pagedown" => "PageDown".into(),
        "shift" => "Shift".into(),
        "control" | "ctrl" => "Control".into(),
        "alt" | "option" => "Alt".into(),
        "meta" | "super" | "command" | "cmd" | "win" => "Meta".into(),
        "capslock" => "CapsLock".into(),
        s if s.len() >= 2 && s.starts_with('f') && s[1..].chars().all(|c| c.is_ascii_digit()) => format!("F{}", &s[1..]),
        _ => if raw.chars().count() == 1 { lower } else { raw },
    }
}
"##;

/// winit 0.28 (simd, Metal, CUDA, ROCm): `VirtualKeyCode` -> canonical key name.
pub(super) const KEY_NAMES_WINIT_028: &str = r##"
#[allow(dead_code)]
fn __boring_vk_name(key: winit::event::VirtualKeyCode) -> String {
    use winit::event::VirtualKeyCode::*;
    let s: &str = match key {
        Escape => "Escape", Space => " ", Return | NumpadEnter => "Enter", Tab => "Tab", Back => "Backspace",
        Delete => "Delete", Insert => "Insert", Left => "Left", Right => "Right", Up => "Up", Down => "Down",
        Home => "Home", End => "End", PageUp => "PageUp", PageDown => "PageDown", Capital => "CapsLock",
        LShift | RShift => "Shift", LControl | RControl => "Control", LAlt | RAlt => "Alt", LWin | RWin => "Meta",
        A => "a", B => "b", C => "c", D => "d", E => "e", F => "f", G => "g", H => "h", I => "i", J => "j",
        K => "k", L => "l", M => "m", N => "n", O => "o", P => "p", Q => "q", R => "r", S => "s", T => "t",
        U => "u", V => "v", W => "w", X => "x", Y => "y", Z => "z",
        Key0 | Numpad0 => "0", Key1 | Numpad1 => "1", Key2 | Numpad2 => "2", Key3 | Numpad3 => "3",
        Key4 | Numpad4 => "4", Key5 | Numpad5 => "5", Key6 | Numpad6 => "6", Key7 | Numpad7 => "7",
        Key8 | Numpad8 => "8", Key9 | Numpad9 => "9",
        F1 => "F1", F2 => "F2", F3 => "F3", F4 => "F4", F5 => "F5", F6 => "F6",
        F7 => "F7", F8 => "F8", F9 => "F9", F10 => "F10", F11 => "F11", F12 => "F12",
        Minus | NumpadSubtract => "-", Equals => "=", Plus | NumpadAdd => "+", Comma | NumpadComma => ",",
        Period | NumpadDecimal => ".", Slash | NumpadDivide => "/", Semicolon => ";", Apostrophe => "'",
        Backslash => "\\", LBracket => "[", RBracket => "]", Grave => "`", Asterisk | NumpadMultiply => "*",
        _ => return format!("{:?}", key),
    };
    s.to_string()
}
"##;

/// winit 0.30 (wgpu): logical `Key` -> canonical key name.
pub(super) const KEY_NAMES_WINIT_030: &str = r##"
#[allow(dead_code)]
fn __boring_logical_key_name(key: &winit::keyboard::Key) -> String {
    use winit::keyboard::{Key, NamedKey};
    match key {
        Key::Character(c) => c.to_lowercase(),
        Key::Named(n) => match n {
            NamedKey::Escape => "Escape".into(),
            NamedKey::Enter => "Enter".into(),
            NamedKey::Space => " ".into(),
            NamedKey::Tab => "Tab".into(),
            NamedKey::Backspace => "Backspace".into(),
            NamedKey::Delete => "Delete".into(),
            NamedKey::Insert => "Insert".into(),
            NamedKey::ArrowLeft => "Left".into(),
            NamedKey::ArrowRight => "Right".into(),
            NamedKey::ArrowUp => "Up".into(),
            NamedKey::ArrowDown => "Down".into(),
            NamedKey::Home => "Home".into(),
            NamedKey::End => "End".into(),
            NamedKey::PageUp => "PageUp".into(),
            NamedKey::PageDown => "PageDown".into(),
            NamedKey::CapsLock => "CapsLock".into(),
            NamedKey::Shift => "Shift".into(),
            NamedKey::Control => "Control".into(),
            NamedKey::Alt => "Alt".into(),
            NamedKey::Super => "Meta".into(),
            other => format!("{:?}", other),
        },
        k => format!("{:?}", k),
    }
}
"##;

/// The type of a lowered `screen.*` input member, so a host can adapt it (`Str` to its own
/// `string` representation, `Int` to a kernel field's width).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum InKind { Bool, Int, Float, Str }

/// `screen.<name>` (a bare property) over the `__BoringInput` reached at `inp`.
pub(super) fn property(name: &str, inp: &str) -> Option<(String, InKind)> {
    use InKind::*;
    Some(match name {
        "text" => (format!("{inp}.text()"), Str),
        "marked" => (format!("{inp}.marked.clone()"), Str),
        "marked_start" => (format!("{inp}.marked_start"), Int),
        "marked_end" => (format!("{inp}.marked_end"), Int),
        "last_key" => (format!("{inp}.last_key.clone()"), Str),
        "shift" => (format!("({inp}.mods & 1 != 0)"), Bool),
        "ctrl" => (format!("({inp}.mods & 2 != 0)"), Bool),
        "alt" => (format!("({inp}.mods & 4 != 0)"), Bool),
        "meta" => (format!("({inp}.mods & 8 != 0)"), Bool),
        "focused" => (format!("{inp}.focused"), Bool),
        "focus_gained" => (format!("{inp}.focus_gained"), Bool),
        "focus_lost" => (format!("{inp}.focus_lost"), Bool),
        _ => return None,
    })
}

/// `screen.mouse.<name>` for the members the input runtime adds (`scroll_x`/`scroll_y`).
pub(super) fn mouse_property(name: &str, inp: &str) -> Option<(String, InKind)> {
    match name {
        "scroll_x" => Some((format!("{inp}.scroll_x"), InKind::Float)),
        "scroll_y" => Some((format!("{inp}.scroll_y"), InKind::Float)),
        _ => None,
    }
}

/// `screen.<method>(arg)`; `arg` is a Rust expression that derefs to `str`.
pub(super) fn method(name: &str, arg: &str, inp: &str) -> Option<String> {
    Some(match name {
        "key_repeat" => format!("{inp}.key_repeat(&*({arg}))"),
        "key_released" => format!("{inp}.key_released(&*({arg}))"),
        "mouse_pressed" => format!("{inp}.mouse_pressed(&*({arg}))"),
        "mouse_released" => format!("{inp}.mouse_released(&*({arg}))"),
        _ => return None,
    })
}

/// Rust that turns a lowered `Str` into the generated crate's `string` (`Arc<str>`).
pub(super) fn str_to_string(expr: &str) -> String { format!("std::sync::Arc::<str>::from({expr})") }

/// The extra `WindowEvent` handlers winit 0.28 hosts need, as `(pattern, body)` pairs over a
/// `WindowEvent` value (the caller wraps them in its own `Event::WindowEvent { event: .. }` arm).
pub(super) fn winit028_arms(inp: &str) -> Vec<(String, Vec<String>)> {
    vec![
        ("winit::event::WindowEvent::ReceivedCharacter(__boring_c)".into(),
         vec![format!("{inp}.typed(__boring_c);")]),
        ("winit::event::WindowEvent::Ime(__boring_ime)".into(),
         vec![
            "match __boring_ime {".into(),
            format!("    winit::event::Ime::Preedit(__boring_s, __boring_cur) => {inp}.ime_preedit(&__boring_s, __boring_cur),"),
            format!("    winit::event::Ime::Commit(__boring_s) => {inp}.ime_commit(&__boring_s),"),
            format!("    _ => {inp}.ime_preedit(\"\", None),"),
            "}".into(),
         ]),
        ("winit::event::WindowEvent::ModifiersChanged(__boring_m)".into(),
         vec![format!("{inp}.set_mods(__boring_m.shift(), __boring_m.ctrl(), __boring_m.alt(), __boring_m.logo());")]),
        ("winit::event::WindowEvent::MouseWheel { delta: __boring_d, .. }".into(),
         vec![
            "match __boring_d {".into(),
            format!("    winit::event::MouseScrollDelta::LineDelta(__boring_x, __boring_y) => {inp}.scroll(__boring_x * 40.0, __boring_y * 40.0),"),
            format!("    winit::event::MouseScrollDelta::PixelDelta(__boring_p) => {inp}.scroll(__boring_p.x as f32, __boring_p.y as f32),"),
            "}".into(),
         ]),
        ("winit::event::WindowEvent::Focused(__boring_f)".into(),
         vec![format!("{inp}.set_focus(__boring_f);")]),
    ]
}

/// A `matches!` pattern list of the 0.28 events that invalidate the window in on-demand mode.
pub(super) const WINIT028_INVALIDATING: &str =
    "winit::event::WindowEvent::ReceivedCharacter(_) | winit::event::WindowEvent::Ime(_) | winit::event::WindowEvent::ModifiersChanged(_) | winit::event::WindowEvent::MouseWheel { .. } | winit::event::WindowEvent::Focused(_)";

/// The `Event::WindowEvent` arm for a winit 0.28 `MouseInput` already exists on every host;
/// the line it must add to feed the shared state (`button` is the winit button value).
pub(super) fn mouse_button_feed(inp: &str, button: &str, down: &str) -> String {
    format!("{inp}.mouse_button(match {button} {{ winit::event::MouseButton::Left => 1, winit::event::MouseButton::Right => 2, winit::event::MouseButton::Middle => 4, _ => 0 }}, {down});")
}

/// The generated crate's `string` type of each parameter of the general-pass function `fname`
/// (`Some("Arc<str>")` / `Some("Rc<str>")`), read back from its signature in `general_code`.
/// Empty when the function is not found. Lets a host whose own string literals are `&str` pass
/// one (or a `screen.text`) to a Boring function that takes a `string`.
pub(super) fn string_params(general_code: &str, fname: &str) -> Vec<Option<&'static str>> {
    let needle = format!("fn {fname}(");
    let Some(sig) = general_code.lines().find_map(|l| {
        let t = l.trim_start();
        let t = t.strip_prefix("pub ").unwrap_or(t);
        t.starts_with(&needle).then(|| t.to_string())
    }) else { return Vec::new() };
    let rest = &sig[needle.len()..];
    let mut depth = 0i32;
    let mut end = None;
    for (i, c) in rest.char_indices() {
        match c {
            '(' | '<' | '[' => depth += 1,
            '>' | ']' => depth -= 1,
            ')' if depth == 0 => { end = Some(i); break; }
            ')' => depth -= 1,
            _ => {}
        }
    }
    let Some(end) = end else { return Vec::new() };
    let params = &rest[..end];
    let (mut out, mut cur, mut depth) = (Vec::new(), String::new(), 0i32);
    let flush = |cur: &mut String, out: &mut Vec<Option<&'static str>>| {
        let t = cur.trim();
        if !t.is_empty() {
            let ty = t.split_once(':').map(|(_, ty)| ty.trim()).unwrap_or("");
            out.push(if ty == "Arc<str>" { Some("Arc<str>") } else if ty == "Rc<str>" { Some("Rc<str>") } else { None });
        }
        cur.clear();
    };
    for c in params.chars() {
        match c {
            '(' | '<' | '[' => { depth += 1; cur.push(c); }
            ')' | '>' | ']' => { depth -= 1; cur.push(c); }
            ',' if depth == 0 => flush(&mut cur, &mut out),
            _ => cur.push(c),
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// `expr` (already lowered) converted to the generated crate's `string` type.
pub(super) fn to_string_type(ty: &str, expr: &str) -> String {
    if ty == "Rc<str>" { format!("std::rc::Rc::<str>::from({expr})") } else { format!("std::sync::Arc::<str>::from({expr})") }
}

/// Whether a loop-body argument lowers to a Rust `&str`/`String` on a winit-0.28 host: a string
/// literal or a string-valued `screen` member (`text`, `marked`, `last_key`).
pub(super) fn is_host_str(e: &crate::ast::Expr, screen_var: Option<&str>, event_vars: &[String]) -> bool {
    use crate::ast::ExprKind;
    match &e.kind {
        ExprKind::Str(_) => true,
        ExprKind::Var(v) => event_vars.iter().any(|e| e == v),
        ExprKind::Field(obj, f) => matches!(&obj.kind, ExprKind::Var(v) if Some(v.as_str()) == screen_var)
            && matches!(f.as_str(), "text" | "marked" | "last_key"),
        _ => false,
    }
}

/// `for k in screen.key_events:` / `screen.key_up_events` -- the frame's key-down (and auto-repeat)
/// / key-up events as canonical names, in arrival order: the name of the `__BoringInput` field
/// holding them. Unlike `key_repeat(k)` this keeps order and duplicates, so a slow frame loses nothing.
pub(super) fn event_list_field(member: &str) -> Option<&'static str> {
    match member { "key_events" => Some("key_repeat"), "key_up_events" => Some("key_released"), _ => None }
}

/// True when `body` contains a `break` that would leave the render loop (not allowed inside an event `for`).
pub(super) fn body_has_break(body: &[crate::ast::Stmt]) -> bool {
    use crate::ast::Stmt;
    body.iter().any(|s| match s {
        Stmt::Break(..) => true,
        Stmt::If(i) => i.branches.iter().any(|(_, b)| body_has_break(b)) || i.else_body.as_ref().is_some_and(|b| body_has_break(b)),
        Stmt::For(f) => body_has_break(&f.body),
        _ => false,
    })
}
