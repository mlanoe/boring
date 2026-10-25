// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Per-kernel backend routing for multi-target builds (`boring build --target all`).
//
// This file is standard-library only, and is used twice: it is copied verbatim into every
// generated multi-target crate as `src/unified.rs`, and compiled into `boring` itself under
// `cfg(test)` so the routing below is unit-tested.
//
// The program itself runs once, on the CPU host code (the simd backend). Each kernel launch
// asks `dispatch` where to run. Other backends register one *adapter* per kernel they can run:
// a function that takes the kernel's fields as `Arg`s (host copies in, host copies out), uploads
// them, launches the backend's own build of the kernel, and writes the results back. `dispatch`
// walks the ranked backends and returns `None` when the CPU should run the kernel itself.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::{Mutex, OnceLock};

pub type BoxError = Box<dyn Error + Send + Sync>;

/// Below this many threads a launch stays on the CPU: uploading, launching and reading back
/// costs more than the computation. `BORING_GPU_MIN_THREADS` overrides it.
const DEFAULT_MIN_THREADS: u64 = 1 << 14;

/// An adapter's way of saying "I cannot run this launch" (for example a block size the backend
/// has fixed at build time). `dispatch` then tries the next backend, and finally the CPU.
#[derive(Debug)]
pub struct Unsupported(pub String);

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unsupported on this backend: {}", self.0)
    }
}

impl Error for Unsupported {}

macro_rules! arg_types {
    ($(($scalar:ident, $buffer:ident, $ty:ty, $get:ident, $get_buf:ident)),* $(,)?) => {
        /// One kernel field: a scalar by value, or a buffer borrowed from the CPU host copy.
        pub enum Arg<'a> {
            $($scalar($ty), $buffer(&'a mut Vec<$ty>),)*
        }

        impl<'a> Arg<'a> {
            $(
                pub fn $get(&self) -> $ty {
                    match self { Arg::$scalar(v) => *v, _ => panic!(concat!("kernel argument is not a ", stringify!($ty))) }
                }
                pub fn $get_buf(&mut self) -> &mut Vec<$ty> {
                    match self { Arg::$buffer(v) => v, _ => panic!(concat!("kernel argument is not a Vec<", stringify!($ty), ">")) }
                }
            )*
        }
    };
}

arg_types!(
    (F32, BufF32, f32, as_f32, buf_f32),
    (F64, BufF64, f64, as_f64, buf_f64),
    (I8, BufI8, i8, as_i8, buf_i8),
    (I16, BufI16, i16, as_i16, buf_i16),
    (I32, BufI32, i32, as_i32, buf_i32),
    (I64, BufI64, i64, as_i64, buf_i64),
    (Isize, BufIsize, isize, as_isize, buf_isize),
    (U8, BufU8, u8, as_u8, buf_u8),
    (U16, BufU16, u16, as_u16, buf_u16),
    (U32, BufU32, u32, as_u32, buf_u32),
    (U64, BufU64, u64, as_u64, buf_u64),
    (Usize, BufUsize, usize, as_usize, buf_usize),
    (Bool, BufBool, bool, as_bool, buf_bool),
);

pub type Adapter = fn(&mut [Arg<'_>], [u32; 3], [u32; 3]) -> Result<(), BoxError>;

/// What one non-CPU backend can run.
pub struct Routes {
    pub name: &'static str,
    /// Creates the backend's device on first use; `false` drops the backend.
    pub init: fn() -> bool,
    /// `(kernel name, adapter)` for every kernel this backend can run.
    pub kernels: Vec<(&'static str, Adapter)>,
}

struct State {
    order: Vec<Routes>,
    forced: bool,
    min_threads: u64,
    /// `BORING_BACKEND_VERBOSE`: report where each kernel runs, when that changes.
    verbose: bool,
    on_use: Box<dyn Fn(&str) + Send + Sync>,
    initialized: Mutex<HashMap<&'static str, bool>>,
    last: Mutex<HashMap<String, &'static str>>,
}

static STATE: OnceLock<State> = OnceLock::new();

/// Register the ranked backends. `forced` is set when the user named the backends
/// (`BORING_BACKEND`): the small-launch rule is then skipped. `on_use` is called the first time
/// a backend runs a kernel (the caller takes its load lease there). A second call is ignored.
pub fn install(order: Vec<Routes>, forced: bool, on_use: Box<dyn Fn(&str) + Send + Sync>) {
    let min_threads = std::env::var("BORING_GPU_MIN_THREADS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(DEFAULT_MIN_THREADS);
    let _ = STATE.set(State {
        order,
        forced,
        min_threads,
        verbose: std::env::var_os("BORING_BACKEND_VERBOSE").is_some(),
        on_use,
        initialized: Mutex::new(HashMap::new()),
        last: Mutex::new(HashMap::new()),
    });
}

/// Run `kernel` on the best backend that can. `None` means: run it on the CPU.
pub fn dispatch(kernel: &str, args: &mut [Arg<'_>], grid: [u32; 3], block: [u32; 3]) -> Option<Result<(), BoxError>> {
    let state = STATE.get()?;
    let threads: u64 = grid.iter().chain(block.iter()).map(|&d| d.max(1) as u64).product();
    if !state.forced && threads < state.min_threads {
        state.record(kernel, "cpu");
        return None;
    }
    for routes in &state.order {
        let Some((_, run)) = routes.kernels.iter().find(|(name, _)| *name == kernel) else { continue };
        if !state.ensure_init(routes) {
            continue;
        }
        match run(args, grid, block) {
            Ok(()) => {
                (state.on_use)(routes.name);
                state.record(kernel, routes.name);
                return Some(Ok(()));
            }
            Err(e) if e.is::<Unsupported>() => continue,
            Err(e) => return Some(Err(e)),
        }
    }
    state.record(kernel, "cpu");
    None
}

/// The backend the most recent launch of `kernel` ran on (`"cpu"` for the CPU).
pub fn last_choice(kernel: &str) -> Option<&'static str> {
    STATE.get()?.last.lock().ok()?.get(kernel).copied()
}

impl State {
    fn ensure_init(&self, routes: &Routes) -> bool {
        let mut done = self.initialized.lock().unwrap();
        *done.entry(routes.name).or_insert_with(|| {
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let ok = std::panic::catch_unwind(routes.init).unwrap_or(false);
            std::panic::set_hook(previous);
            ok
        })
    }

    fn record(&self, kernel: &str, backend: &'static str) {
        if let Ok(mut last) = self.last.lock() {
            let changed = last.insert(kernel.to_string(), backend) != Some(backend);
            if changed && self.verbose {
                eprintln!("boring: kernel {kernel} -> {backend}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn double(args: &mut [Arg<'_>], _grid: [u32; 3], _block: [u32; 3]) -> Result<(), BoxError> {
        let factor = args[0].as_f32();
        for v in args[1].buf_f32().iter_mut() {
            *v *= factor;
        }
        Ok(())
    }

    fn refuses(_: &mut [Arg<'_>], _: [u32; 3], _: [u32; 3]) -> Result<(), BoxError> {
        Err(Box::new(Unsupported("fixed block size".into())))
    }

    fn fails(_: &mut [Arg<'_>], _: [u32; 3], _: [u32; 3]) -> Result<(), BoxError> {
        Err("device lost".into())
    }

    fn up() -> bool {
        true
    }

    fn down() -> bool {
        false
    }

    // The registry is process-global, so every behaviour is exercised from one test.
    #[test]
    fn routes_launches_across_ranked_backends() {
        let used = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&used);
        install(
            vec![
                Routes { name: "broken", init: down, kernels: vec![("Scale", double)] },
                Routes { name: "picky", init: up, kernels: vec![("Scale", refuses), ("Other", double)] },
                Routes { name: "gpu", init: up, kernels: vec![("Scale", double), ("Boom", fails)] },
            ],
            false,
            Box::new(move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
            }),
        );

        // A small launch stays on the CPU even though a backend could run it.
        let mut data = vec![1.0f32, 2.0];
        let mut args = [Arg::F32(3.0), Arg::BufF32(&mut data)];
        assert!(dispatch("Scale", &mut args, [1, 1, 1], [2, 1, 1]).is_none());
        assert_eq!(last_choice("Scale"), Some("cpu"));
        assert_eq!(data, vec![1.0, 2.0], "the CPU path is the caller's to run");

        // A large launch skips the backend whose device is unavailable, skips the one that
        // refuses this launch, and runs on the next.
        let mut data = vec![1.0f32, 2.0];
        let mut args = [Arg::F32(3.0), Arg::BufF32(&mut data)];
        let ran = dispatch("Scale", &mut args, [1024, 1, 1], [256, 1, 1]);
        assert!(matches!(ran, Some(Ok(()))));
        assert_eq!(data, vec![3.0, 6.0]);
        assert_eq!(last_choice("Scale"), Some("gpu"));
        assert_eq!(used.load(Ordering::SeqCst), 1);

        // A kernel no backend knows about goes to the CPU.
        let mut args: [Arg; 0] = [];
        assert!(dispatch("Unknown", &mut args, [1024, 1, 1], [256, 1, 1]).is_none());

        // A real failure is reported, not retried elsewhere.
        let mut args: [Arg; 0] = [];
        let failed = dispatch("Boom", &mut args, [1024, 1, 1], [256, 1, 1]);
        assert!(matches!(failed, Some(Err(e)) if e.to_string().contains("device lost")));
    }
}
