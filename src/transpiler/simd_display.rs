// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// GPU-display runtime for `--target simd` (docs/gpu-display.md): the generated
// crate's `Dimension`/`Screen` types and the winit 0.28 + softbuffer 0.3 software
// blit behind `screen.present(...)`.
//
// The user's `kernel: loop:` compiles to an ordinary Rust `loop` (see
// `emit_stmt.rs`'s `KernelBlock` arm), so it cannot live inside a winit event
// handler the way the CUDA/ROCm hosts' `run_return` closure does. Instead
// `fn main()` runs the user program on a worker thread and keeps the OS event
// loop on the main thread (a hard requirement on macOS); `Screen::present`
// hands each frame across and blocks until the main thread has blitted it.

/// `Dimension` (host) and `__DeviceDimension` (kernel field mirror, device widths),
/// plus the `'surface` seeding helper. Emitted whenever a program uses either.
pub(super) const DIMENSION_RUNTIME: &str = r#"
#[allow(dead_code)] #[derive(Clone, Copy, Debug, Default, PartialEq)] struct Dimension { width: isize, height: isize }
#[allow(dead_code)] impl Dimension { fn new(width: isize, height: isize) -> Self { Dimension { width, height } } }
#[allow(dead_code)] #[derive(Clone, Copy, Debug, Default, PartialEq)] struct __DeviceDimension { width: i32, height: i32 }
// A Dimension-sized kernel's `'surface` buffers start as sparse random 0/1 cells
// (~30% set), the same seed the wgpu Screen host writes (examples/game_of_life.br).
#[allow(dead_code)] fn __boring_surface_seed(n: usize) -> Vec<u32> {
    let mut rng: u64 = 0x12345678ABCDEF01;
    (0..n).map(|_| { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; if rng % 10 < 3 { 1u32 } else { 0u32 } }).collect()
}
"#;

/// `Screen` and its windowing thread. Emitted only when the program constructs one.
pub(super) const SCREEN_RUNTIME: &str = r#"
// ---- GPU display: `Screen` (software blit: winit 0.28 + softbuffer 0.3) ----
#[allow(dead_code)]
enum __BoringCmd {
    Create { width: u32, height: u32, title: String, done: std::sync::mpsc::Sender<()> },
    Present,
    ImeArea { x: i32, y: i32, w: u32, h: u32 },
    Exit,
}
struct __BoringDisplay {
    proxy: std::sync::Mutex<Option<winit::event_loop::EventLoopProxy<__BoringCmd>>>,
    keys: std::sync::Mutex<std::collections::HashSet<String>>,
    // Keys that went from released to pressed since the last `Screen::present` returned.
    just_pressed: std::sync::Mutex<std::collections::HashSet<String>>,
    mouse_x: std::sync::atomic::AtomicI64,
    mouse_y: std::sync::atomic::AtomicI64,
    // Bit 0 = left, 1 = right, 2 = middle.
    mouse_buttons: std::sync::atomic::AtomicU8,
    // Typed text, IME composition, modifiers, focus, wheel, key repeat/release, button edges.
    input: std::sync::Mutex<__BoringInput>,
    // Current window size in pixels.
    size: (std::sync::atomic::AtomicU32, std::sync::atomic::AtomicU32),
    frame: std::sync::Mutex<Option<Vec<u32>>>,
    // (width, height) of the presented frame; follows a kernel reallocated after a resize.
    frame_dims: std::sync::Mutex<(usize, usize)>,
    frame_taken: std::sync::Condvar,
    closed: std::sync::atomic::AtomicBool,
    resized: std::sync::atomic::AtomicBool,
    // Set by input/resize/close events; `Screen::present` waits on it in on-demand mode.
    dirty: std::sync::Mutex<bool>,
    dirty_cv: std::sync::Condvar,
}
impl __BoringDisplay {
    fn invalidate(&self) { *self.dirty.lock().unwrap() = true; self.dirty_cv.notify_all(); }
}
fn __boring_display() -> &'static __BoringDisplay {
    static DISPLAY: std::sync::OnceLock<__BoringDisplay> = std::sync::OnceLock::new();
    DISPLAY.get_or_init(|| __BoringDisplay {
        proxy: std::sync::Mutex::new(None),
        keys: std::sync::Mutex::new(Default::default()),
        just_pressed: std::sync::Mutex::new(Default::default()),
        mouse_x: std::sync::atomic::AtomicI64::new(0),
        mouse_y: std::sync::atomic::AtomicI64::new(0),
        mouse_buttons: std::sync::atomic::AtomicU8::new(0),
        input: std::sync::Mutex::new(__BoringInput::new()),
        size: (std::sync::atomic::AtomicU32::new(0), std::sync::atomic::AtomicU32::new(0)),
        frame: std::sync::Mutex::new(None),
        frame_dims: std::sync::Mutex::new((0, 0)),
        frame_taken: std::sync::Condvar::new(),
        closed: std::sync::atomic::AtomicBool::new(false),
        resized: std::sync::atomic::AtomicBool::new(false),
        dirty: std::sync::Mutex::new(false),
        dirty_cv: std::sync::Condvar::new(),
    })
}
fn __boring_send(cmd: __BoringCmd) {
    let proxy = __boring_display().proxy.lock().unwrap();
    proxy.as_ref().expect("Screen needs the display event loop started by `main`").send_event(cmd).ok();
}

// `screen.mouse`: cursor position in pixel-buffer coordinates (the window's pixel count equals
// the buffer's) and button state, snapshotted once per `Screen::present`.
#[allow(dead_code)] #[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Mouse { x: isize, y: isize, left: bool, right: bool, middle: bool, scroll_x: f32, scroll_y: f32 }

#[allow(dead_code)]
struct Screen {
    mouse: Mouse,
    pressed: std::collections::HashSet<String>,
    input: __BoringInput,
    // `screen.text` & co.: the frame's input snapshot (see docs/gpu-display.md, "Input").
    text: std::sync::Arc<str>,
    marked: std::sync::Arc<str>,
    last_key: std::sync::Arc<str>,
    // `for k in screen.key_events:` / `screen.key_up_events`: this frame's key events in order.
    key_events: Vec<std::sync::Arc<str>>,
    key_up_events: Vec<std::sync::Arc<str>>,
    marked_start: isize,
    marked_end: isize,
    shift: bool,
    ctrl: bool,
    alt: bool,
    meta: bool,
    focused: bool,
    focus_gained: bool,
    focus_lost: bool,
    dimension: Dimension,
    width: isize,
    height: isize,
    resized: bool,
    closed: bool,
    frame: isize,
    time: f64,
    start: std::time::Instant,
    redraw_requested: bool,
}
#[allow(dead_code)]
impl Screen {
    fn new<T: AsRef<str>>(dimension: Dimension, title: T) -> Self {
        let (done, wait) = std::sync::mpsc::channel();
        __boring_send(__BoringCmd::Create {
            width: dimension.width.max(1) as u32,
            height: dimension.height.max(1) as u32,
            title: title.as_ref().to_string(),
            done,
        });
        wait.recv().expect("display thread closed before the window was created");
        Screen {
            mouse: Mouse::default(),
            pressed: Default::default(),
            input: __BoringInput::new(),
            text: "".into(),
            marked: "".into(),
            last_key: "".into(),
            key_events: Vec::new(),
            key_up_events: Vec::new(),
            marked_start: -1,
            marked_end: -1,
            shift: false,
            ctrl: false,
            alt: false,
            meta: false,
            focused: true,
            focus_gained: false,
            focus_lost: false,
            dimension,
            width: dimension.width,
            height: dimension.height,
            resized: false,
            closed: false,
            frame: 0,
            time: 0.0,
            start: std::time::Instant::now(),
            redraw_requested: false,
        }
    }
    // On-demand mode (`Screen(..., redraw = "demand")`): run the next frame without waiting
    // for an input/resize event. A no-op in the default continuous mode.
    // (`screen.requestRedraw()` in Boring source; method names are snake_cased on this target.)
    fn request_redraw(&mut self) { self.redraw_requested = true; }
    // `screen.setImeArea(x, y, w, h)`: where the input method may put its candidate window, in window
    // (= buffer) pixels.
    fn set_ime_area(&self, x: isize, y: isize, w: isize, h: isize) {
        __boring_send(__BoringCmd::ImeArea { x: x as i32, y: y as i32, w: w.max(0) as u32, h: h.max(0) as u32 });
    }
    // `"\x1B"` reaches here either as the ESC character or as its 4-character escape text.
    fn key<T: AsRef<str>>(&self, key: T) -> bool {
        __boring_display().keys.lock().unwrap().contains(&__boring_key_name(key.as_ref()))
    }
    // True only for the frame in which `key` went from released to pressed (auto-repeat excluded).
    fn key_pressed<T: AsRef<str>>(&self, key: T) -> bool {
        self.pressed.contains(&__boring_key_name(key.as_ref()))
    }
    // True on the frame `key` went down and on every frame an OS auto-repeat of it arrived.
    fn key_repeat<T: AsRef<str>>(&self, key: T) -> bool { self.input.key_repeat(key.as_ref()) }
    // True on the frame `key` was released.
    fn key_released<T: AsRef<str>>(&self, key: T) -> bool { self.input.key_released(key.as_ref()) }
    // True on the frame mouse button "left" / "right" / "middle" went down (up).
    fn mouse_pressed<T: AsRef<str>>(&self, button: T) -> bool { self.input.mouse_pressed(button.as_ref()) }
    fn mouse_released<T: AsRef<str>>(&self, button: T) -> bool { self.input.mouse_released(button.as_ref()) }
    fn present(&mut self, pixels: &[u32]) {
        use std::sync::atomic::Ordering::SeqCst;
        let display = __boring_display();
        if display.closed.load(SeqCst) { std::process::exit(0); }
        // A buffer allocated for the window's current size (`k = K(Dimension(screen.width, screen.height))`
        // after a resize) is `width * height` long; otherwise it is the size the window opened with.
        let dims = if pixels.len() == (self.width.max(0) * self.height.max(0)) as usize { (self.width as usize, self.height as usize) }
            else { (self.dimension.width as usize, self.dimension.height as usize) };
        *display.frame_dims.lock().unwrap() = dims;
        *display.frame.lock().unwrap() = Some(pixels.to_vec());
        __boring_send(__BoringCmd::Present);
        let mut slot = display.frame.lock().unwrap();
        while slot.is_some() && !display.closed.load(SeqCst) {
            slot = display.frame_taken.wait(slot).unwrap();
        }
        drop(slot);
        if display.closed.load(SeqCst) { std::process::exit(0); }
        if __boring_redraw_on_demand() {
            // Sleep until something invalidates the window (or the body asked for a frame).
            let mut dirty = display.dirty.lock().unwrap();
            while !*dirty && !self.redraw_requested && !display.closed.load(SeqCst) {
                dirty = display.dirty_cv.wait(dirty).unwrap();
            }
            *dirty = false;
            drop(dirty);
            self.redraw_requested = false;
            if display.closed.load(SeqCst) { std::process::exit(0); }
        }
        self.frame += 1;
        self.time = self.start.elapsed().as_secs_f64();
        self.resized = display.resized.swap(false, SeqCst);
        self.closed = false;
        self.pressed = std::mem::take(&mut *display.just_pressed.lock().unwrap());
        let (ww, wh) = (display.size.0.load(SeqCst), display.size.1.load(SeqCst));
        if ww > 0 && wh > 0 { self.width = ww as isize; self.height = wh as isize; }
        self.input = display.input.lock().unwrap().snapshot();
        self.text = self.input.text().into();
        self.marked = self.input.marked.as_str().into();
        self.last_key = self.input.last_key.as_str().into();
        self.key_events = self.input.key_repeat.iter().map(|k| k.as_str().into()).collect();
        self.key_up_events = self.input.key_released.iter().map(|k| k.as_str().into()).collect();
        self.marked_start = self.input.marked_start;
        self.marked_end = self.input.marked_end;
        self.shift = self.input.mods & 1 != 0;
        self.ctrl = self.input.mods & 2 != 0;
        self.alt = self.input.mods & 4 != 0;
        self.meta = self.input.mods & 8 != 0;
        self.focused = self.input.focused;
        self.focus_gained = self.input.focus_gained;
        self.focus_lost = self.input.focus_lost;
        let buttons = display.mouse_buttons.load(SeqCst);
        self.mouse = Mouse {
            x: display.mouse_x.load(SeqCst) as isize,
            y: display.mouse_y.load(SeqCst) as isize,
            left: buttons & 1 != 0,
            right: buttons & 2 != 0,
            middle: buttons & 4 != 0,
            scroll_x: self.input.scroll_x,
            scroll_y: self.input.scroll_y,
        };
    }
}

// Runs `program` on a worker thread and serves the window on this (main) thread.
fn __boring_display_run<F: FnOnce() + Send + 'static>(program: F) {
    use std::sync::atomic::Ordering::SeqCst;
    use winit::event::{ElementState, Event, WindowEvent};
    use winit::event_loop::ControlFlow;
    use winit::platform::run_return::EventLoopExtRunReturn;
    let display = __boring_display();
    let mut event_loop = winit::event_loop::EventLoopBuilder::<__BoringCmd>::with_user_event().build();
    *display.proxy.lock().unwrap() = Some(event_loop.create_proxy());
    struct Done(winit::event_loop::EventLoopProxy<__BoringCmd>);
    impl Drop for Done { fn drop(&mut self) { let _ = self.0.send_event(__BoringCmd::Exit); } }
    let done = Done(event_loop.create_proxy());
    let worker = std::thread::spawn(move || { let _done = done; program(); });
    let mut window: Option<winit::window::Window> = None;
    let mut blit: Option<(softbuffer::Context, softbuffer::Surface)> = None;
    let mut frame_size = (0usize, 0usize);
    let mut win_size = (0u32, 0u32);
    let mut surface_size = (0u32, 0u32);
    event_loop.run_return(|event, target, flow| {
        *flow = ControlFlow::Wait;
        match event {
            Event::UserEvent(__BoringCmd::Create { width, height, title, done }) => {
                // `PhysicalSize`: the window's pixel count must equal the pixel buffer's
                // regardless of the display's DPI scale (see docs/gpu-display.md).
                let built = winit::window::WindowBuilder::new()
                    .with_title(title)
                    .with_inner_size(winit::dpi::PhysicalSize::new(width, height))
                    .build(target)
                    .expect("failed to create the display window");
                // The window outlives the surface: both are dropped together at the end of this fn.
                let context = unsafe { softbuffer::Context::new(&built) }.expect("softbuffer context");
                let surface = unsafe { softbuffer::Surface::new(&context, &built) }.expect("softbuffer surface");
                frame_size = (width as usize, height as usize);
                win_size = (width, height);
                display.size.0.store(width, SeqCst);
                display.size.1.store(height, SeqCst);
                built.set_ime_allowed(true);
                window = Some(built);
                blit = Some((context, surface));
                let _ = done.send(());
            }
            Event::UserEvent(__BoringCmd::Present) => {
                let pixels = display.frame.lock().unwrap().take();
                { let d = *display.frame_dims.lock().unwrap(); if d.0 > 0 { frame_size = d; } }
                if let (Some(pixels), Some((_, surface))) = (pixels, blit.as_mut()) {
                    let (ww, wh) = win_size;
                    if let (Some(nw), Some(nh)) = (std::num::NonZeroU32::new(ww), std::num::NonZeroU32::new(wh)) {
                        if surface_size != win_size {
                            let _ = surface.resize(nw, nh);
                            surface_size = win_size;
                        }
                        if let Ok(mut buf) = surface.buffer_mut() {
                            let (fw, ww) = (frame_size.0, ww as usize);
                            let rows = (pixels.len() / fw.max(1)).min(frame_size.1).min(wh as usize);
                            let n = fw.min(ww);
                            for y in 0..rows {
                                buf[y * ww..y * ww + n].copy_from_slice(&pixels[y * fw..y * fw + n]);
                            }
                            let _ = buf.present();
                        }
                    }
                }
                display.frame_taken.notify_all();
            }
            Event::UserEvent(__BoringCmd::ImeArea { x, y, w, h }) => {
                if let Some(window) = &window {
                    window.set_ime_position(winit::dpi::PhysicalPosition::new(x, y + h as i32));
                    let _ = w;
                }
            }
            Event::UserEvent(__BoringCmd::Exit) => { *flow = ControlFlow::Exit; }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                display.closed.store(true, SeqCst);
                display.frame_taken.notify_all();
                display.invalidate();
                *flow = ControlFlow::Exit;
            }
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                win_size = (size.width, size.height);
                display.size.0.store(size.width, SeqCst);
                display.size.1.store(size.height, SeqCst);
                display.resized.store(true, SeqCst);
                display.invalidate();
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
                if let Some(key) = input.virtual_keycode {
                    let name = __boring_vk_name(key);
                    let mut keys = display.keys.lock().unwrap();
                    match input.state {
                        ElementState::Pressed => {
                            display.input.lock().unwrap().key_down(&name);
                            // `insert` is false for an auto-repeat of a key already held.
                            if keys.insert(name.clone()) { display.just_pressed.lock().unwrap().insert(name); }
                        }
                        ElementState::Released => { keys.remove(&name); display.input.lock().unwrap().key_up(&name); }
                    }
                    display.invalidate();
                }
            }
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                display.mouse_x.store(position.x.floor() as i64, SeqCst);
                display.mouse_y.store(position.y.floor() as i64, SeqCst);
                display.invalidate();
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state, button, .. }, .. } => {
                let bit = match button {
                    winit::event::MouseButton::Left => 1u8,
                    winit::event::MouseButton::Right => 2,
                    winit::event::MouseButton::Middle => 4,
                    _ => 0,
                };
                display.input.lock().unwrap().mouse_button(bit, state == ElementState::Pressed);
                match state {
                    ElementState::Pressed => { display.mouse_buttons.fetch_or(bit, SeqCst); }
                    ElementState::Released => { display.mouse_buttons.fetch_and(!bit, SeqCst); }
                }
                display.invalidate();
            }
            Event::WindowEvent { event: WindowEvent::ReceivedCharacter(c), .. } => {
                display.input.lock().unwrap().typed(c);
                display.invalidate();
            }
            Event::WindowEvent { event: WindowEvent::Ime(ime), .. } => {
                {
                    let mut input = display.input.lock().unwrap();
                    match ime {
                        winit::event::Ime::Preedit(text, cursor) => input.ime_preedit(&text, cursor),
                        winit::event::Ime::Commit(text) => input.ime_commit(&text),
                        _ => input.ime_preedit("", None),
                    }
                }
                display.invalidate();
            }
            Event::WindowEvent { event: WindowEvent::ModifiersChanged(m), .. } => {
                display.input.lock().unwrap().set_mods(m.shift(), m.ctrl(), m.alt(), m.logo());
                display.invalidate();
            }
            Event::WindowEvent { event: WindowEvent::MouseWheel { delta, .. }, .. } => {
                {
                    let mut input = display.input.lock().unwrap();
                    match delta {
                        winit::event::MouseScrollDelta::LineDelta(x, y) => input.scroll(x * 40.0, y * 40.0),
                        winit::event::MouseScrollDelta::PixelDelta(p) => input.scroll(p.x as f32, p.y as f32),
                    }
                }
                display.invalidate();
            }
            Event::WindowEvent { event: WindowEvent::Focused(focused), .. } => {
                display.input.lock().unwrap().set_focus(focused);
                display.invalidate();
            }
            _ => {}
        }
    });
    let _ = &window;
    if display.closed.load(SeqCst) { std::process::exit(0); }
    if worker.join().is_err() { std::process::exit(101); }
}
"#;
