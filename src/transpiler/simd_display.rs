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
    Exit,
}
struct __BoringDisplay {
    proxy: std::sync::Mutex<Option<winit::event_loop::EventLoopProxy<__BoringCmd>>>,
    keys: std::sync::Mutex<std::collections::HashSet<String>>,
    frame: std::sync::Mutex<Option<Vec<u32>>>,
    frame_taken: std::sync::Condvar,
    closed: std::sync::atomic::AtomicBool,
    resized: std::sync::atomic::AtomicBool,
}
fn __boring_display() -> &'static __BoringDisplay {
    static DISPLAY: std::sync::OnceLock<__BoringDisplay> = std::sync::OnceLock::new();
    DISPLAY.get_or_init(|| __BoringDisplay {
        proxy: std::sync::Mutex::new(None),
        keys: std::sync::Mutex::new(Default::default()),
        frame: std::sync::Mutex::new(None),
        frame_taken: std::sync::Condvar::new(),
        closed: std::sync::atomic::AtomicBool::new(false),
        resized: std::sync::atomic::AtomicBool::new(false),
    })
}
fn __boring_send(cmd: __BoringCmd) {
    let proxy = __boring_display().proxy.lock().unwrap();
    proxy.as_ref().expect("Screen needs the display event loop started by `main`").send_event(cmd).ok();
}

#[allow(dead_code)]
struct Screen {
    dimension: Dimension,
    width: isize,
    height: isize,
    resized: bool,
    closed: bool,
    frame: isize,
    time: f64,
    start: std::time::Instant,
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
            dimension,
            width: dimension.width,
            height: dimension.height,
            resized: false,
            closed: false,
            frame: 0,
            time: 0.0,
            start: std::time::Instant::now(),
        }
    }
    // `"\x1B"` reaches here either as the ESC character or as its 4-character escape text.
    fn key(&self, key: &str) -> bool {
        let key = match key { "\\x1B" | "\\x1b" => "\x1B", "\\r" => "\r", "\\n" => "\n", other => other };
        __boring_display().keys.lock().unwrap().contains(key)
    }
    fn present(&mut self, pixels: &[u32]) {
        use std::sync::atomic::Ordering::SeqCst;
        let display = __boring_display();
        if display.closed.load(SeqCst) { std::process::exit(0); }
        *display.frame.lock().unwrap() = Some(pixels.to_vec());
        __boring_send(__BoringCmd::Present);
        let mut slot = display.frame.lock().unwrap();
        while slot.is_some() && !display.closed.load(SeqCst) {
            slot = display.frame_taken.wait(slot).unwrap();
        }
        drop(slot);
        if display.closed.load(SeqCst) { std::process::exit(0); }
        self.frame += 1;
        self.time = self.start.elapsed().as_secs_f64();
        self.resized = display.resized.swap(false, SeqCst);
        self.closed = false;
    }
}

fn __boring_key_str(key: winit::event::VirtualKeyCode) -> String {
    use winit::event::VirtualKeyCode::*;
    match key {
        Escape => "\x1B".into(),
        Space => " ".into(),
        Return => "\r".into(),
        Up => "\x1B[A".into(),
        Down => "\x1B[B".into(),
        Left => "\x1B[D".into(),
        Right => "\x1B[C".into(),
        A => "a".into(), B => "b".into(), C => "c".into(),
        D => "d".into(), E => "e".into(), F => "f".into(),
        G => "g".into(), H => "h".into(), I => "i".into(),
        J => "j".into(), K => "k".into(), L => "l".into(),
        M => "m".into(), N => "n".into(), O => "o".into(),
        P => "p".into(), Q => "q".into(), R => "r".into(),
        S => "s".into(), T => "t".into(), U => "u".into(),
        V => "v".into(), W => "w".into(), X => "x".into(),
        Y => "y".into(), Z => "z".into(),
        Key1 => "1".into(), Key2 => "2".into(), Key3 => "3".into(),
        Key4 => "4".into(), Key5 => "5".into(), Key6 => "6".into(),
        Key7 => "7".into(), Key8 => "8".into(), Key9 => "9".into(),
        Key0 => "0".into(),
        _ => format!("{:?}", key).to_lowercase(),
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
                window = Some(built);
                blit = Some((context, surface));
                let _ = done.send(());
            }
            Event::UserEvent(__BoringCmd::Present) => {
                let pixels = display.frame.lock().unwrap().take();
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
            Event::UserEvent(__BoringCmd::Exit) => { *flow = ControlFlow::Exit; }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                display.closed.store(true, SeqCst);
                display.frame_taken.notify_all();
                *flow = ControlFlow::Exit;
            }
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                win_size = (size.width, size.height);
                display.resized.store(true, SeqCst);
            }
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
                if let Some(key) = input.virtual_keycode {
                    let name = __boring_key_str(key);
                    let mut keys = display.keys.lock().unwrap();
                    match input.state {
                        ElementState::Pressed => { keys.insert(name); }
                        ElementState::Released => { keys.remove(&name); }
                    }
                }
            }
            _ => {}
        }
    });
    let _ = &window;
    if display.closed.load(SeqCst) { std::process::exit(0); }
    if worker.join().is_err() { std::process::exit(101); }
}
"#;
