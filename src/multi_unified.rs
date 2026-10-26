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
// asks `dispatch` where to run. Other backends register one *route* per kernel they can run:
// `run` takes the kernel's fields as `Arg`s, uploads the buffers it is told to, and launches the
// backend's own build of the kernel; `read` copies one buffer back; `release` frees the
// instance. `dispatch` walks the ranked backends and returns `None` when the CPU should run the
// kernel itself.
//
// Buffers stay on the device between launches. The router tracks, per kernel instance and per
// argument, whether the device copy is valid (so an unchanged input is uploaded once) and whether
// it is newer than the CPU copy (so a result is only copied back when the CPU needs it: it reads
// the field, launches the kernel on the CPU, or the instance moves to another backend).

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

pub type BoxError = Box<dyn Error + Send + Sync>;

/// A class of execution device, for placement hints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Cpu,
    /// Any GPU.
    Gpu,
    /// A GPU with its own memory.
    Discrete,
    /// A GPU sharing the CPU's memory.
    Integrated,
}

impl Class {
    fn name(self) -> &'static str {
        match self {
            Class::Cpu => "cpu",
            Class::Gpu => "gpu",
            Class::Discrete => "discrete",
            Class::Integrated => "integrated",
        }
    }

    /// Does a backend of class `have` (`"discrete"`, `"integrated"`, `"gpu"` when unknown or
    /// `"cpu"`) belong to this class?
    fn admits(self, have: &str) -> bool {
        match self {
            Class::Cpu => have == "cpu",
            Class::Gpu => have != "cpu",
            Class::Discrete => have == "discrete",
            Class::Integrated => have == "integrated",
        }
    }
}

/// Where a launch may run. `prefer` is a wish (`@prefer(.gpu)` on the kernel, `prefer = .cpu`
/// on the launch, which wins); `only` is a requirement (`@only(.gpu)`) that fails the launch
/// when it cannot be met.
#[derive(Clone, Copy, Debug, Default)]
pub struct Hints {
    pub prefer: Option<Class>,
    pub only: Option<Class>,
}

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
    (F16, BufF16, half::f16, as_f16, buf_f16),
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

/// What a kernel argument is, for the residency bookkeeping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Scalar,
    /// A buffer the kernel only reads.
    Buffer,
    /// A buffer the kernel writes.
    MutBuffer,
}

/// A routable kernel as the CPU host describes it: its name and the kind of each argument, in
/// the order of the `Arg` slice it passes to `dispatch`.
pub struct Spec {
    pub name: &'static str,
    pub args: &'static [Kind],
}

/// Run a launch. `upload[i]` is true when argument `i` is a buffer whose CPU copy must be
/// uploaded first; for the others the device already holds the current data. Scalars are always
/// set from `args`. Must not touch the device state before returning `Unsupported`.
pub type Run = fn(u64, &mut [Arg<'_>], &[bool], [u32; 3], [u32; 3]) -> Result<(), BoxError>;
/// Copy buffer argument `index` of instance `id` from the device into `out`.
pub type Read = fn(u64, usize, &mut Arg<'_>) -> Result<(), BoxError>;
/// Free the device side of instance `id`.
pub type Release = fn(u64);

pub struct KernelRoute {
    pub name: &'static str,
    pub run: Run,
    pub read: Read,
    pub release: Release,
}

/// What `kernel.backends()` reports for one backend.
#[derive(Clone, Debug)]
pub struct BackendInfo {
    pub name: &'static str,
    /// `"gpu"` or `"cpu"`.
    pub kind: &'static str,
    /// `"discrete"`, `"integrated"`, `"gpu"` when a GPU's memory model is unknown, or `"cpu"`.
    pub class: &'static str,
    /// The start-up benchmark score, or the prior for a backend that cannot be measured.
    pub score: f64,
    /// How many other programs were using this backend at start-up.
    pub load: usize,
}

/// What one non-CPU backend can run.
pub struct Routes {
    pub name: &'static str,
    /// Creates the backend's device on first use; `false` drops the backend.
    pub init: fn() -> bool,
    pub kernels: Vec<KernelRoute>,
}

/// Where the data of one kernel instance lives.
struct Instance {
    kernel: &'static str,
    /// The backend holding the instance's device copies.
    backend: Option<&'static str>,
    /// Per argument: the device copy is newer than the CPU copy.
    remote_newest: Vec<bool>,
    /// Per argument: the device copy matches the newest data.
    device_valid: Vec<bool>,
}

struct State {
    order: Vec<Routes>,
    forced: bool,
    min_threads: u64,
    /// `BORING_BACKEND_VERBOSE`: report where each kernel runs, when that changes.
    verbose: bool,
    on_use: Box<dyn Fn(&str) + Send + Sync>,
    infos: Vec<BackendInfo>,
    initialized: Mutex<HashMap<&'static str, bool>>,
    last: Mutex<HashMap<String, &'static str>>,
    instances: Mutex<HashMap<u64, Instance>>,
}

static STATE: OnceLock<State> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static UPLOADS: AtomicU64 = AtomicU64::new(0);
static DOWNLOADS: AtomicU64 = AtomicU64::new(0);

/// The backends this program may run kernels on, best first, the CPU among them.
pub fn backends() -> Vec<BackendInfo> {
    match STATE.get() {
        Some(state) => state.infos.clone(),
        None => vec![BackendInfo { name: "cpu", kind: "cpu", class: "cpu", score: 0.0, load: 0 }],
    }
}

/// A fresh id for a kernel instance on the CPU host.
pub fn new_instance() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// How many buffers were uploaded to / downloaded from a backend so far.
pub fn transfers() -> (u64, u64) {
    (UPLOADS.load(Ordering::Relaxed), DOWNLOADS.load(Ordering::Relaxed))
}

/// Wraps a backend's per-instance handles so they can live in a `static`. A GPU API handle is
/// used from whichever thread launches the kernel, one at a time behind the store's lock.
pub struct AssertSend<T>(pub T);

unsafe impl<T> Send for AssertSend<T> {}

/// Register the ranked backends. `forced` is set when the user named the backends
/// (`BORING_BACKEND`): the small-launch rule is then skipped. `on_use` is called the first time
/// a backend runs a kernel (the caller takes its load lease there). A second call is ignored.
pub fn install(order: Vec<Routes>, forced: bool, infos: Vec<BackendInfo>, on_use: Box<dyn Fn(&str) + Send + Sync>) {
    let min_threads = std::env::var("BORING_GPU_MIN_THREADS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(DEFAULT_MIN_THREADS);
    let _ = STATE.set(State {
        order,
        forced,
        min_threads,
        verbose: std::env::var_os("BORING_BACKEND_VERBOSE").is_some(),
        on_use,
        infos,
        initialized: Mutex::new(HashMap::new()),
        last: Mutex::new(HashMap::new()),
        instances: Mutex::new(HashMap::new()),
    });
}

/// Run `spec`'s kernel on the best backend that can. `None` means: run it on the CPU (its
/// buffers are then current on the CPU).
///
/// - `only = Cpu`: always the CPU. `only = Gpu`: a backend must take it, else the launch fails.
/// - `prefer = Cpu`: the CPU, unless the user forced backends with `BORING_BACKEND`.
/// - `prefer = Gpu` (or `only = Gpu`): the small-launch rule does not apply.
pub fn dispatch(spec: &Spec, id: u64, args: &mut [Arg<'_>], grid: [u32; 3], block: [u32; 3], hints: Hints) -> Option<Result<(), BoxError>> {
    let kernel = spec.name;
    let Some(state) = STATE.get() else {
        // Not installed (no backend usable): the CPU runs everything.
        return match hints.only {
            Some(Class::Cpu) | None => None,
            Some(class) => Some(Err(Box::new(NoBackend::new(kernel, class)))),
        };
    };
    if hints.only == Some(Class::Cpu) || (hints.prefer == Some(Class::Cpu) && hints.only.is_none() && !state.forced) {
        return state.on_cpu(spec, id, args);
    }
    let required = hints.only.filter(|c| *c != Class::Cpu);
    // `BORING_BACKEND` overrides a wish, not a requirement.
    let wished = if state.forced { None } else { hints.prefer.filter(|c| *c != Class::Cpu) };
    let wants_gpu = required.is_some() || wished.is_some();
    let threads: u64 = grid.iter().chain(block.iter()).map(|&d| d.max(1) as u64).product();
    if !state.forced && !wants_gpu && threads < state.min_threads {
        return state.on_cpu(spec, id, args);
    }
    // A requirement or a wish for one kind of GPU (`.discrete`, `.integrated`) is tried first;
    // a wish that no backend meets falls back to any backend, a requirement does not.
    let wanted = required.or(wished);
    let mut passes = vec![wanted];
    if wanted.is_some() && required.is_none() && wanted != Some(Class::Gpu) {
        passes.push(None);
    }
    for pass in passes {
        for routes in &state.order {
            if let Some(class) = pass {
                if !class.admits(state.class_of(routes.name)) {
                    continue;
                }
            }
            let Some(route) = routes.kernels.iter().find(|r| r.name == kernel) else { continue };
            if !state.ensure_init(routes) {
                continue;
            }
            match state.launch_on(routes, route, spec, id, args, grid, block) {
                Ok(()) => {
                    (state.on_use)(routes.name);
                    state.record(kernel, routes.name);
                    return Some(Ok(()));
                }
                Err(e) if e.is::<Unsupported>() => continue,
                Err(e) => return Some(Err(e)),
            }
        }
    }
    if let Some(class) = required {
        return Some(Err(Box::new(NoBackend::new(kernel, class))));
    }
    state.on_cpu(spec, id, args)
}

/// Is the newest data of buffer argument `index` of instance `id` on a device, not on the CPU?
pub fn is_remote(id: u64, index: usize) -> bool {
    let Some(state) = STATE.get() else { return false };
    let instances = state.instances.lock().unwrap();
    instances.get(&id).is_some_and(|i| i.remote_newest.get(index).copied().unwrap_or(false))
}

/// The CPU host reads buffer `index` of instance `id`: copy the device's newer data into `out`
/// (a copy of the CPU buffer). Returns false, leaving `out` alone, when the CPU copy is current.
pub fn fetch(id: u64, index: usize, out: &mut Arg<'_>) -> bool {
    let Some(state) = STATE.get() else { return false };
    let instances = state.instances.lock().unwrap();
    let Some(inst) = instances.get(&id) else { return false };
    if !inst.remote_newest.get(index).copied().unwrap_or(false) {
        return false;
    }
    let Some(read) = state.read_of(inst.backend, inst.kernel) else { return false };
    DOWNLOADS.fetch_add(1, Ordering::Relaxed);
    read(id, index, out).is_ok()
}

/// The CPU host is about to replace buffer `index` of instance `id` entirely: the device copy
/// is stale from now on.
pub fn invalidate(id: u64, index: usize) {
    let Some(state) = STATE.get() else { return };
    let mut instances = state.instances.lock().unwrap();
    if let Some(inst) = instances.get_mut(&id) {
        if index < inst.device_valid.len() {
            inst.device_valid[index] = false;
            inst.remote_newest[index] = false;
        }
    }
}

/// The CPU host is about to modify buffer `index` of instance `id` in place: bring the device's
/// newer data into `out` first (the CPU buffer itself), then mark the device copy stale.
pub fn prepare_write(id: u64, index: usize, out: &mut Arg<'_>) {
    let Some(state) = STATE.get() else { return };
    let mut instances = state.instances.lock().unwrap();
    let Some(inst) = instances.get_mut(&id) else { return };
    if index >= inst.device_valid.len() {
        return;
    }
    if inst.remote_newest[index] {
        if let Some(read) = state.read_of(inst.backend, inst.kernel) {
            DOWNLOADS.fetch_add(1, Ordering::Relaxed);
            let _ = read(id, index, out);
        }
    }
    inst.device_valid[index] = false;
    inst.remote_newest[index] = false;
}

/// The CPU host drops instance `id`: free its device side.
pub fn release(id: u64) {
    let Some(state) = STATE.get() else { return };
    let removed = state.instances.lock().unwrap().remove(&id);
    if let Some(inst) = removed {
        if let Some(backend) = inst.backend {
            if let Some(route) = state.route_of(backend, inst.kernel) {
                (route.release)(id);
            }
        }
    }
}

/// `@only(.gpu)` could not be met.
#[derive(Debug)]
pub struct NoBackend(String);

impl NoBackend {
    fn new(kernel: &str, class: Class) -> NoBackend {
        NoBackend(format!("kernel `{kernel}` is `@only(.{})` but no matching backend could run this launch", class.name()))
    }
}

impl fmt::Display for NoBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for NoBackend {}

/// The backend the most recent launch of `kernel` ran on (`"cpu"` for the CPU).
pub fn last_choice(kernel: &str) -> Option<&'static str> {
    STATE.get()?.last.lock().ok()?.get(kernel).copied()
}

impl State {
    /// The class of the backend called `name` (`"gpu"` when it is not listed).
    fn class_of(&self, name: &str) -> &'static str {
        self.infos.iter().find(|i| i.name == name).map_or("gpu", |i| i.class)
    }

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

    fn route_of(&self, backend: &str, kernel: &str) -> Option<&KernelRoute> {
        self.order.iter().find(|r| r.name == backend)?.kernels.iter().find(|k| k.name == kernel)
    }

    fn read_of(&self, backend: Option<&'static str>, kernel: &str) -> Option<Read> {
        Some(self.route_of(backend?, kernel)?.read)
    }

    /// Copy every argument whose newest data is on the instance's device back to the CPU.
    fn pull_remote(&self, inst: &mut Instance, id: u64, args: &mut [Arg<'_>]) -> Result<(), BoxError> {
        let Some(read) = self.read_of(inst.backend, inst.kernel) else { return Ok(()) };
        for (i, arg) in args.iter_mut().enumerate() {
            if inst.remote_newest.get(i).copied().unwrap_or(false) {
                DOWNLOADS.fetch_add(1, Ordering::Relaxed);
                read(id, i, arg)?;
                inst.remote_newest[i] = false;
            }
        }
        Ok(())
    }

    /// The CPU runs this launch: its buffers must be current on the CPU, and what it writes
    /// makes the device copies stale.
    fn on_cpu(&self, spec: &Spec, id: u64, args: &mut [Arg<'_>]) -> Option<Result<(), BoxError>> {
        let mut instances = self.instances.lock().unwrap();
        if let Some(inst) = instances.get_mut(&id) {
            if let Err(e) = self.pull_remote(inst, id, args) {
                return Some(Err(e));
            }
            for (i, kind) in spec.args.iter().enumerate() {
                if *kind == Kind::MutBuffer {
                    inst.device_valid[i] = false;
                }
            }
        }
        drop(instances);
        self.record(spec.name, "cpu");
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_on(&self, routes: &Routes, route: &KernelRoute, spec: &Spec, id: u64, args: &mut [Arg<'_>], grid: [u32; 3], block: [u32; 3]) -> Result<(), BoxError> {
        let mut instances = self.instances.lock().unwrap();
        let inst = instances.entry(id).or_insert_with(|| Instance {
            kernel: spec.name,
            backend: None,
            remote_newest: vec![false; spec.args.len()],
            device_valid: vec![false; spec.args.len()],
        });
        // The instance moves between backends: the old device's newer data comes back first.
        if let Some(previous) = inst.backend {
            if previous != routes.name {
                self.pull_remote(inst, id, args)?;
                if let Some(old) = self.route_of(previous, inst.kernel) {
                    (old.release)(id);
                }
                inst.backend = None;
                inst.device_valid.iter_mut().for_each(|v| *v = false);
            }
        }
        let upload: Vec<bool> = spec.args.iter().enumerate().map(|(i, kind)| *kind != Kind::Scalar && !inst.device_valid[i]).collect();
        (route.run)(id, args, &upload, grid, block)?;
        UPLOADS.fetch_add(upload.iter().filter(|u| **u).count() as u64, Ordering::Relaxed);
        for (i, kind) in spec.args.iter().enumerate() {
            match kind {
                Kind::Scalar => {}
                Kind::Buffer => inst.device_valid[i] = true,
                Kind::MutBuffer => {
                    inst.device_valid[i] = true;
                    inst.remote_newest[i] = true;
                }
            }
        }
        inst.backend = Some(routes.name);
        Ok(())
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

    /// A fake device memory per backend: `data = data * factor + bias[0]`, with `factor == 99`
    /// refused by backend "a".
    static DEVICE: Mutex<Option<HashMap<(&'static str, u64, usize), Vec<f32>>>> = Mutex::new(None);

    fn with_device<T>(f: impl FnOnce(&mut HashMap<(&'static str, u64, usize), Vec<f32>>) -> T) -> T {
        f(DEVICE.lock().unwrap().get_or_insert_with(HashMap::new))
    }

    fn run_on(backend: &'static str, id: u64, args: &mut [Arg<'_>], upload: &[bool]) -> Result<(), BoxError> {
        let factor = args[0].as_f32();
        if backend == "a" && factor == 99.0 {
            return Err(Box::new(Unsupported("factor 99".into())));
        }
        for i in 1..3 {
            if upload[i] {
                let copy = args[i].buf_f32().clone();
                with_device(|d| d.insert((backend, id, i), copy));
            }
        }
        with_device(|d| {
            let bias = d[&(backend, id, 1)][0];
            for v in d.get_mut(&(backend, id, 2)).unwrap().iter_mut() {
                *v = *v * factor + bias;
            }
        });
        Ok(())
    }

    fn read_from(backend: &'static str, id: u64, index: usize, out: &mut Arg<'_>) -> Result<(), BoxError> {
        *out.buf_f32() = with_device(|d| d[&(backend, id, index)].clone());
        Ok(())
    }

    fn release_on(backend: &'static str, id: u64) {
        with_device(|d| d.retain(|(b, i, _), _| !(*b == backend && *i == id)));
    }

    fn run_a(id: u64, args: &mut [Arg<'_>], upload: &[bool], _: [u32; 3], _: [u32; 3]) -> Result<(), BoxError> {
        run_on("a", id, args, upload)
    }
    fn run_b(id: u64, args: &mut [Arg<'_>], upload: &[bool], _: [u32; 3], _: [u32; 3]) -> Result<(), BoxError> {
        run_on("b", id, args, upload)
    }
    fn read_a(id: u64, index: usize, out: &mut Arg<'_>) -> Result<(), BoxError> {
        read_from("a", id, index, out)
    }
    fn read_b(id: u64, index: usize, out: &mut Arg<'_>) -> Result<(), BoxError> {
        read_from("b", id, index, out)
    }
    fn release_a(id: u64) {
        release_on("a", id)
    }
    fn release_b(id: u64) {
        release_on("b", id)
    }
    fn up() -> bool {
        true
    }

    const SPEC: Spec = Spec { name: "Affine", args: &[Kind::Scalar, Kind::Buffer, Kind::MutBuffer] };

    fn launch(id: u64, factor: f32, bias: &mut Vec<f32>, data: &mut Vec<f32>, hints: Hints) -> Option<Result<(), BoxError>> {
        let mut args = [Arg::F32(factor), Arg::BufF32(bias), Arg::BufF32(data)];
        dispatch(&SPEC, id, &mut args, [1024, 1, 1], [256, 1, 1], hints)
    }

    // The registry is process-global, so every behaviour is exercised from one test.
    #[test]
    fn buffers_stay_on_the_device_between_launches() {
        install(
            vec![
                Routes { name: "a", init: up, kernels: vec![KernelRoute { name: "Affine", run: run_a, read: read_a, release: release_a }] },
                Routes { name: "b", init: up, kernels: vec![KernelRoute { name: "Affine", run: run_b, read: read_b, release: release_b }] },
            ],
            true,
            vec![BackendInfo { name: "a", kind: "gpu", class: "discrete", score: 2.0, load: 0 }, BackendInfo { name: "cpu", kind: "cpu", class: "cpu", score: 1.0, load: 1 }],
            Box::new(|_| {}),
        );
        assert_eq!(backends().iter().map(|b| (b.name, b.kind, b.load)).collect::<Vec<_>>(), [("a", "gpu", 0), ("cpu", "cpu", 1)]);
        let id = new_instance();
        let (mut bias, mut data) = (vec![10.0f32], vec![1.0f32, 2.0]);
        let uploaded = || transfers().0;
        let downloaded = || transfers().1;
        let (up0, down0) = (uploaded(), downloaded());

        // First launch uploads both buffers; the result stays on the device.
        assert!(matches!(launch(id, 2.0, &mut bias, &mut data, Hints::default()), Some(Ok(()))));
        assert_eq!((uploaded() - up0, downloaded() - down0), (2, 0));
        assert_eq!(data, vec![1.0, 2.0], "the CPU copy is stale until it is read");
        assert!(is_remote(id, 2) && !is_remote(id, 1));
        assert_eq!(last_choice("Affine"), Some("a"));

        // The second launch uploads nothing: bias and data are already there.
        assert!(matches!(launch(id, 3.0, &mut bias, &mut data, Hints::default()), Some(Ok(()))));
        assert_eq!(uploaded() - up0, 2);

        // The CPU reads the field: it gets the device's data, and the device keeps it.
        let mut seen = data.clone();
        assert!(fetch(id, 2, &mut Arg::BufF32(&mut seen)));
        assert_eq!(seen, vec![46.0, 52.0]);
        assert_eq!(downloaded() - down0, 1);
        assert!(is_remote(id, 2));
        assert!(!fetch(id, 1, &mut Arg::BufF32(&mut seen)), "the bias is current on the CPU");

        // The CPU replaces the buffer: only that buffer is uploaded next.
        invalidate(id, 2);
        assert!(!is_remote(id, 2));
        data = vec![0.0, 0.0];
        assert!(matches!(launch(id, 1.0, &mut bias, &mut data, Hints::default()), Some(Ok(()))));
        assert_eq!(uploaded() - up0, 3);

        // The CPU modifies it in place: the device's data comes first, then the copy is stale.
        let mut working = data.clone();
        prepare_write(id, 2, &mut Arg::BufF32(&mut working));
        assert_eq!(working, vec![10.0, 10.0]);
        data = working;
        assert!(matches!(launch(id, 2.0, &mut bias, &mut data, Hints::default()), Some(Ok(()))));
        assert_eq!(uploaded() - up0, 4);
        assert_eq!(with_device(|d| d[&("a", id, 2)].clone()), vec![30.0, 30.0]);

        // A launch kept on the CPU brings the device's data back and leaves the copy stale.
        let cpu_only = Hints { prefer: None, only: Some(Class::Cpu) };
        assert!(launch(id, 1.0, &mut bias, &mut data, cpu_only).is_none());
        assert_eq!(data, vec![30.0, 30.0]);
        assert!(!is_remote(id, 2));
        assert_eq!(last_choice("Affine"), Some("cpu"));
        assert!(matches!(launch(id, 1.0, &mut bias, &mut data, Hints::default()), Some(Ok(()))));
        assert_eq!(uploaded() - up0, 5, "only the buffer the CPU might have changed");

        // Backend "a" refuses a launch: the instance moves to "b", data first, everything uploaded.
        assert!(matches!(launch(id, 99.0, &mut bias, &mut data, Hints::default()), Some(Ok(()))));
        assert_eq!(last_choice("Affine"), Some("b"));
        assert_eq!(data, vec![40.0, 40.0], "pulled from a before moving");
        assert_eq!(uploaded() - up0, 7);
        assert!(with_device(|d| d.keys().all(|(b, i, _)| !(*b == "a" && *i == id))), "a released the instance");

        // Dropping the instance frees the device side.
        release(id);
        assert!(with_device(|d| d.keys().all(|(_, i, _)| *i != id)));
        assert!(!is_remote(id, 2));
    }
}
