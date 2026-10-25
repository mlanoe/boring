// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `boring build --target all` / `--target cuda,metal`: one Cargo crate holding several backends.
//
// Every backend's transpiler still produces what it always did (a host `main.rs` plus its device
// source). Here those host files become modules (`src/backend_<name>.rs`, each behind a Cargo
// feature of the same name) and the device sources keep their per-backend names
// (`kernels/main.metal`, `kernels/main.cu`, `kernels/main.hip`, `shaders/main.wgsl`), so they sit
// side by side without collision.
//
// The program itself runs once, on the CPU host (the simd backend). Each kernel launch goes
// through `unified::dispatch` (`src/multi_unified.rs`), which hands it to the best backend that
// has an adapter for that kernel and keeps small launches on the CPU. The generated `main.rs`
// ranks the usable backends at start-up (`src/multi_policy.rs`) and installs them there.

use std::collections::BTreeMap;

use crate::transpiler::unified::RouteField;
use std::path::{Path, PathBuf};
use std::process;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Backend {
    Cuda,
    Rocm,
    Metal,
    Wgpu,
    Simd,
}

/// Default preference order at run time: native vendor backends, then the portable GPU one,
/// then the CPU. (A performance-driven order is layered on later.)
pub const PREFERENCE: [Backend; 5] = [Backend::Cuda, Backend::Rocm, Backend::Metal, Backend::Wgpu, Backend::Simd];

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Rocm => "rocm",
            Backend::Metal => "metal",
            Backend::Wgpu => "wgpu",
            Backend::Simd => "simd",
        }
    }

    pub fn from_name(name: &str) -> Option<Backend> {
        PREFERENCE.into_iter().find(|b| b.name() == name)
    }

    /// Why this backend cannot be built on the current host, or `None` if it can.
    fn unavailable_reason(self) -> Option<String> {
        match self {
            Backend::Metal if !cfg!(target_os = "macos") => Some("Metal needs macOS".into()),
            Backend::Cuda if cfg!(target_os = "macos") => Some("CUDA is not available on macOS".into()),
            Backend::Cuda if !on_path("nvcc") => Some("`nvcc` not found on PATH".into()),
            Backend::Rocm if cfg!(target_os = "macos") => Some("ROCm is not available on macOS".into()),
            Backend::Rocm if !on_path("hipcc") && !rocm_path_has_hipcc() => Some("`hipcc` not found".into()),
            _ => None,
        }
    }
}

fn on_path(program: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else { return false };
    std::env::split_paths(&path).any(|dir| {
        dir.join(program).is_file() || dir.join(format!("{program}.exe")).is_file() || dir.join(format!("{program}.bat")).is_file()
    })
}

fn rocm_path_has_hipcc() -> bool {
    let root = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    Path::new(&root).join("bin").join("hipcc").is_file() || Path::new(&root).join("bin").join("hipcc.bin.exe").is_file()
}

/// Parse one `--target` value: a backend name, `all`, or a comma-separated list of those.
/// Returns the requested backends in preference order, without duplicates, plus the backends
/// `all` skipped because this host cannot build them (with the reason, for a note).
pub type Skipped = Vec<(Backend, String)>;

pub fn parse_targets(value: &str) -> Result<(Vec<Backend>, Skipped), String> {
    let mut wanted: Vec<Backend> = Vec::new();
    let mut skipped: Vec<(Backend, String)> = Vec::new();
    for part in value.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        if part == "all" {
            for backend in PREFERENCE {
                match backend.unavailable_reason() {
                    None => wanted.push(backend),
                    Some(reason) => skipped.push((backend, reason)),
                }
            }
        } else if let Some(backend) = Backend::from_name(part) {
            wanted.push(backend);
        } else {
            return Err(format!("unknown target '{part}'"));
        }
    }
    if wanted.is_empty() {
        return Err("--target requires a value".into());
    }
    wanted.sort();
    wanted.dedup();
    skipped.retain(|(b, _)| !wanted.contains(b));
    skipped.sort();
    skipped.dedup();
    Ok((wanted, skipped))
}

/// What one backend's transpiler produced, ready to be written into the combined crate.
struct Unit {
    backend: Backend,
    host_rs: String,
    /// Device sources, relative to the project directory.
    files: Vec<(PathBuf, String)>,
    cargo_toml: String,
    build_rs: Option<String>,
    /// Routed builds: the kernels this backend can run for the CPU host, with their arguments.
    routes: Vec<(String, Vec<RouteField>)>,
}

pub fn emit_multi(path: &str, version: &str, backends: &[Backend], skipped: &[(Backend, String)]) -> PathBuf {
    crate::install_kernel_target_project_config(Path::new(path));
    let program = crate::parse_and_merge_program(path);
    let path = PathBuf::from(path);

    let source = std::fs::read_to_string(&path).unwrap_or_default();
    if crate::report_check_result(&path, &source, crate::checker::check_kernel_dispatch_only(&program)) {
        process::exit(1);
    }

    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "output".to_string());
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let project_dir = base_dir.join(format!("{stem}_multi"));
    let tensor_config = crate::tensor_linear_config_for_source(&path);

    // The program runs once, on the CPU host (simd), which routes each kernel launch to a
    // backend (see `src/multi_unified.rs`): the CPU host always takes part.
    let mut backends = backends.to_vec();
    if !backends.contains(&Backend::Simd) {
        backends.push(Backend::Simd);
        backends.sort();
    }
    let backends = &backends[..];

    let mut units: Vec<Unit> = Vec::new();
    let mut kernel_names: Vec<String> = Vec::new();
    for &backend in backends {
        let (unit, names, errors) = match backend {
            Backend::Cuda => {
                let out = crate::transpiler::cuda::transpile_cuda_unified(&program, &stem, version, &tensor_config);
                let unit = Unit {
                    backend,
                    host_rs: out.host_rs,
                    files: vec![(PathBuf::from("kernels/main.cu"), out.device_cu)],
                    cargo_toml: crate::kernel_target_cargo_toml(&out.cargo_toml, false, &path),
                    build_rs: Some(out.build_rs),
                    routes: out.routes,
                };
                (unit, out.kernel_names, out.errors)
            }
            Backend::Rocm => {
                let out = crate::transpiler::rocm::transpile_rocm_unified(&program, &stem, version, &tensor_config);
                let unit = Unit {
                    backend,
                    host_rs: out.host_rs,
                    files: vec![(PathBuf::from("kernels/main.hip"), out.device_hip)],
                    cargo_toml: crate::kernel_target_cargo_toml(&out.cargo_toml, false, &path),
                    build_rs: Some(out.build_rs),
                    routes: out.routes,
                };
                (unit, out.kernel_names, out.errors)
            }
            Backend::Metal => {
                let out = crate::transpiler::metal::transpile_metal_unified(&program, &stem, version, &tensor_config);
                let unit = Unit {
                    backend,
                    host_rs: out.host_rs,
                    files: vec![(PathBuf::from("kernels/main.metal"), out.device_msl)],
                    cargo_toml: crate::kernel_target_cargo_toml(&out.cargo_toml, false, &path),
                    build_rs: None,
                    routes: out.routes,
                };
                (unit, out.kernel_names, out.errors)
            }
            Backend::Wgpu => {
                let out = crate::transpiler::wgpu::transpile_wgpu_unified(&program, &stem, version, &tensor_config);
                let mut files = vec![(PathBuf::from("shaders/main.wgsl"), out.device_wgsl)];
                if let Some(emulated) = out.device_wgsl_emulated {
                    files.push((PathBuf::from("shaders/main_emulated.wgsl"), emulated));
                }
                let unit = Unit {
                    backend,
                    host_rs: out.host_rs,
                    files,
                    cargo_toml: crate::kernel_target_cargo_toml(&out.cargo_toml, out.uses_unicode, &path),
                    build_rs: None,
                    routes: out.routes,
                };
                (unit, out.kernel_names, out.errors)
            }
            Backend::Simd => {
                let out = crate::transpiler::simd::transpile_simd_unified(&program, &stem, version);
                let unit = Unit {
                    backend,
                    host_rs: out.host_rs,
                    files: Vec::new(),
                    cargo_toml: crate::kernel_target_cargo_toml(&out.cargo_toml, out.uses_unicode, &path),
                    build_rs: None,
                    routes: out.routes,
                };
                (unit, out.kernel_names, out.errors)
            }
        };
        if !errors.is_empty() && backend != Backend::Simd {
            eprintln!("warning: the {} backend cannot compile this program and is left out:", backend.name());
            for e in &errors {
                eprintln!("  line {}: {}", e.line, e.message);
            }
            continue;
        }
        if !errors.is_empty() {
            eprintln!("error: transpiling for the {} backend failed", backend.name());
            crate::report_transpile_errors(&path, &source, &errors);
            process::exit(1);
        }
        if kernel_names.is_empty() {
            kernel_names = names;
        }
        units.push(unit);
    }

    // Routed build: an adapter is usable only when the CPU host and the backend agree on the
    // kernel's argument list (each may lower the program slightly differently).
    let mut routed_kernels: Vec<(Backend, Vec<String>)> = Vec::new();
    {
        let cpu: BTreeMap<String, Vec<RouteField>> =
            units.iter().find(|u| u.backend == Backend::Simd).map(|u| u.routes.iter().cloned().collect()).unwrap_or_default();
        for unit in &units {
            if unit.backend == Backend::Simd {
                continue;
            }
            let covered: Vec<String> =
                unit.routes.iter().filter(|(name, fields)| cpu.get(name) == Some(fields)).map(|(name, _)| name.clone()).collect();
            routed_kernels.push((unit.backend, covered));
        }
    }

    let manifest = combined_manifest(&stem, version, &units);

    let src_dir = project_dir.join("src");
    if let Err(e) = std::fs::create_dir_all(&src_dir) {
        fail_write(&src_dir, e);
    }
    let write = |rel: &Path, contents: &str| {
        let target = project_dir.join(rel);
        if let Some(parent) = target.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                fail_write(parent, e);
            }
        }
        if let Err(e) = std::fs::write(&target, contents) {
            fail_write(&target, e);
        }
    };

    for unit in &units {
        let no_aliases = Vec::new();
        let aliases = manifest.aliases.get(&unit.backend).unwrap_or(&no_aliases);
        let module = module_source(unit.backend, &unit.host_rs, aliases).unwrap_or_else(|e| {
            eprintln!("error: {} backend: {e}", unit.backend.name());
            process::exit(1);
        });
        write(&PathBuf::from(format!("src/backend_{}.rs", unit.backend.name())), &module);
        for (rel, contents) in &unit.files {
            write(rel, contents);
        }
    }
    let present: Vec<Backend> = units.iter().map(|u| u.backend).collect();
    write(Path::new("src/main.rs"), &root_source(&present, &routed_kernels));
    write(Path::new("src/unified.rs"), include_str!("multi_unified.rs"));
    write(Path::new("src/policy.rs"), include_str!("multi_policy.rs"));
    write(Path::new("Cargo.toml"), &manifest.text);
    if let Some(build_rs) = combined_build_rs(&units) {
        write(Path::new("build.rs"), &build_rs);
    }

    let names: Vec<&str> = backends.iter().map(|b| b.name()).collect();
    eprintln!("Generated multi-target project at '{}'", project_dir.display());
    eprintln!("  Backends: {}", names.join(", "));
    for (backend, reason) in skipped {
        eprintln!("  Skipped {}: {}", backend.name(), reason);
    }
    eprintln!("  cd {} && cargo build --release", project_dir.display());
    eprintln!("  Each kernel launch picks its backend at run time; BORING_BACKEND=<name>[,<name>...] forces an order, BORING_BACKEND=list shows availability.");
    if !kernel_names.is_empty() {
        eprintln!("  Kernels: {}", kernel_names.join(", "));
    }
    project_dir
}

fn fail_write(path: &Path, e: std::io::Error) -> ! {
    eprintln!("error: cannot write '{}': {}", path.display(), e);
    process::exit(1);
}

/// Turn a backend's standalone `main.rs` into a module: `fn main` becomes a public entry point,
/// a crate-root-only `#[macro_use] extern crate` moves to the root (see `root_source`), and an
/// availability probe plus a uniform `__boring_entry` are appended. `aliases` re-imports crates
/// renamed in the manifest (see `combined_manifest`) under the name the generated code uses.
fn module_source(backend: Backend, host_rs: &str, aliases: &[(String, String)]) -> Result<String, String> {
    let mut out = String::with_capacity(host_rs.len() + 512);
    let mut renamed = 0;
    for line in host_rs.lines() {
        if line.trim() == "#[macro_use] extern crate objc;" {
            continue;
        }
        if let Some(rest) = line.strip_prefix("fn main(") {
            out.push_str("pub fn __boring_backend_main(");
            out.push_str(rest);
            renamed += 1;
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    if renamed != 1 {
        return Err(format!("expected exactly one top-level `fn main`, found {renamed}"));
    }
    let entry = match backend {
        Backend::Cuda | Backend::Rocm | Backend::Metal => "__boring_backend_main()",
        Backend::Wgpu | Backend::Simd => "{ __boring_backend_main(); Ok(()) }",
    };
    out.push_str(&format!(
        "\n// ---- multi-target glue ----\n\
         {}\n\
         {}\n\
         pub fn __boring_entry() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{ {entry} }}\n",
        crate::multi_bench::device_name_fn(backend),
        crate::multi_bench::bench_fn(backend),
    ));
    for (key, ident) in aliases {
        out.push_str(&format!("use {key} as {ident};\n"));
    }
    Ok(out)
}

fn root_source(backends: &[Backend], routed: &[(Backend, Vec<String>)]) -> String {
    let mut s = String::new();
    s.push_str("// Generated by `boring build --target ...` (multi-target). Backends are modules behind\n");
    s.push_str("// Cargo features of the same name; `policy` ranks the usable ones at start-up.\n");
    s.push_str("#![allow(dead_code, unused_variables, unused_assignments, unused_mut, unused_imports, unused_parens, non_snake_case, non_camel_case_types, non_upper_case_globals, unexpected_cfgs, private_interfaces)]\n\n");
    s.push_str("mod policy;\n");
    s.push_str("mod unified;\n");
    for b in backends {
        let n = b.name();
        if *b == Backend::Metal {
            s.push_str("#[cfg(feature = \"metal\")]\n#[macro_use]\nextern crate objc;\n");
        }
        s.push_str(&format!("#[cfg(feature = \"{n}\")]\nmod backend_{n};\n"));
    }
    s.push_str(
        r#"
struct Backend {
    name: &'static str,
    device_name: fn() -> Option<String>,
    bench: fn() -> Option<f64>,
    run: fn() -> Result<(), Box<dyn std::error::Error + Send + Sync>>,
}

/// Compiled-in backends, in static preference order.
fn compiled_backends() -> Vec<Backend> {
    #[allow(unused_mut)]
    let mut all: Vec<Backend> = Vec::new();
"#,
    );
    for b in backends {
        let n = b.name();
        s.push_str(&format!(
            "    #[cfg(feature = \"{n}\")]\n    all.push(Backend {{ name: \"{n}\", device_name: backend_{n}::__boring_device_name, bench: backend_{n}::__boring_bench, run: backend_{n}::__boring_entry }});\n"
        ));
    }
    s.push_str(
        r#"    all
}

/// Run `f` without letting a failing driver load print a panic message; a panic is `None`.
fn silently<T>(f: impl FnOnce() -> Option<T> + std::panic::UnwindSafe) -> Option<T> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f).ok().flatten();
    std::panic::set_hook(previous);
    result
}

fn verbose() -> bool {
    std::env::var_os("BORING_BACKEND_VERBOSE").is_some()
}

/// Order the usable backends by measured speed and current load (see `policy.rs`).
/// `BORING_BENCH=off` keeps the static order, `BORING_BENCH=force` measures again.
fn rank<'a>(usable: &[(&'a Backend, String)], dir: Option<&std::path::Path>) -> Vec<&'a Backend> {
    let mode = std::env::var("BORING_BENCH").unwrap_or_default().trim().to_lowercase();
    if mode == "off" || usable.len() < 2 {
        return usable.iter().map(|(backend, _)| *backend).collect();
    }
    let mut cache = dir.map(policy::load_cache).unwrap_or_default();
    let now = policy::now();
    let mut measured_now = false;
    let mut candidates = Vec::new();
    for (backend, device) in usable {
        let cached = cache.get(backend.name).filter(|entry| mode != "force" && entry.is_fresh(device, now)).map(|entry| entry.score);
        let score = match cached {
            Some(score) => score,
            None => {
                if !measured_now {
                    eprintln!("boring: benchmarking the available backends (cached afterwards)");
                    measured_now = true;
                }
                let score = silently(backend.bench);
                cache.insert(backend.name.to_string(), policy::Entry { device: device.clone(), score, stamp: now });
                score
            }
        };
        let load = dir.map(|d| policy::current_load(d, backend.name)).unwrap_or(0);
        candidates.push((backend.name.to_string(), score, load));
    }
    if measured_now {
        if let Some(dir) = dir {
            policy::save_cache(dir, &cache);
        }
    }
    let ranked = policy::rank(&candidates);
    if verbose() {
        for r in &ranked {
            eprintln!(
                "boring: {:<6} score {:>10.0}{} load {} -> {:.0}",
                r.name, r.score, if r.measured { "" } else { " (prior)" }, r.load, r.effective
            );
        }
    }
    ranked.iter().filter_map(|r| usable.iter().map(|(b, _)| *b).find(|b| b.name == r.name)).collect()
}
"#,
    );
    let mut arms = String::new();
    for (backend, kernels) in routed {
        let n = backend.name();
        let list: Vec<String> = kernels.iter().map(|k| format!("(\"{k}\", backend_{n}::__boring_unified_{k} as unified::Adapter)")).collect();
        arms.push_str(&format!(
            "        #[cfg(feature = \"{n}\")]\n        \"{n}\" => Some(unified::Routes {{ name: \"{n}\", init: backend_{n}::__boring_unified_init, kernels: vec![{}] }}),\n",
            list.join(", ")
        ));
    }
    s.push_str(&ROUTED_MAIN.replace("ROUTES_ARMS", &arms));
    s
}

const ROUTED_MAIN: &str = r#"
/// Backends that run kernels for the CPU host (see `unified.rs`), by name.
fn routes_for(name: &str) -> Option<unified::Routes> {
    match name {
ROUTES_ARMS        _ => None,
    }
}

/// Load leases taken on first use of a backend, held until the process exits.
static LEASES: std::sync::Mutex<Vec<(String, policy::Lease)>> = std::sync::Mutex::new(Vec::new());

fn take_lease(dir: &Option<std::path::PathBuf>, name: &str) {
    let Some(dir) = dir else { return };
    let mut leases = LEASES.lock().unwrap();
    if !leases.iter().any(|(held, _)| held == name) {
        leases.push((name.to_string(), policy::acquire(dir, name)));
    }
}

fn main() {
    let compiled = compiled_backends();
    let requested = std::env::var("BORING_BACKEND").unwrap_or_default();
    let requested = requested.trim().to_lowercase();
    let dir = policy::cache_dir();

    if requested == "list" {
        let cache = dir.as_deref().map(policy::load_cache).unwrap_or_default();
        for backend in &compiled {
            match silently(backend.device_name) {
                Some(device) => {
                    let score = cache.get(backend.name).and_then(|e| e.score).map(|s| format!(", score {s:.0}")).unwrap_or_default();
                    println!("{}: available ({device}{score})", backend.name);
                }
                None => println!("{}: unavailable", backend.name),
            }
        }
        return;
    }

    // An explicit BORING_BACKEND list is the user's own order: no benchmark, and launches are not
    // kept on the CPU for being small. The CPU (`simd`) runs whatever no listed backend can.
    let forced = !requested.is_empty();
    let wanted: Vec<&Backend> = if forced {
        let mut chosen = Vec::new();
        for name in requested.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            match compiled.iter().find(|b| b.name == name) {
                Some(backend) => chosen.push(backend),
                None => eprintln!("boring: backend '{name}' is not compiled into this binary (skipped)"),
            }
        }
        chosen
    } else {
        compiled.iter().collect()
    };
    let usable: Vec<(&Backend, String)> = wanted.into_iter().filter_map(|b| silently(b.device_name).map(|d| (b, d))).collect();
    let ranked: Vec<&Backend> = if forced { usable.iter().map(|(b, _)| *b).collect() } else { rank(&usable, dir.as_deref()) };

    // Backends ranked above the CPU get the kernels they can run; the CPU takes the rest.
    let order: Vec<unified::Routes> = ranked.iter().take_while(|b| b.name != "simd").filter_map(|b| routes_for(b.name)).collect();
    if verbose() {
        let names: Vec<&str> = order.iter().map(|r| r.name).collect();
        eprintln!("boring: kernels run on [{}], else on the CPU", names.join(", "));
    }
    take_lease(&dir, "simd");
    let lease_dir = dir.clone();
    unified::install(order, forced, Box::new(move |name| take_lease(&lease_dir, name)));

    if let Err(e) = backend_simd::__boring_entry() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
"#;

/// `build.rs` running each GPU toolchain step only for the features that are enabled.
fn combined_build_rs(units: &[Unit]) -> Option<String> {
    let scripts: Vec<(&Unit, &String)> = units.iter().filter_map(|u| u.build_rs.as_ref().map(|b| (u, b))).collect();
    if scripts.is_empty() {
        return None;
    }
    let mut s = String::from("// Generated by `boring build --target ...` (multi-target): one section per backend that needs\n// a toolchain step, run only when that backend's Cargo feature is enabled.\n\n");
    for (unit, script) in &scripts {
        let n = unit.backend.name();
        s.push_str(&format!("mod build_{n} {{\n"));
        for line in script.lines() {
            // The backend's own `fn main` becomes this module's entry point.
            if line == "fn main() {" {
                s.push_str("pub fn run() {\n");
            } else {
                s.push_str(line);
                s.push('\n');
            }
        }
        s.push_str("}\n\n");
    }
    s.push_str("fn main() {\n");
    for (unit, _) in &scripts {
        let n = unit.backend.name();
        s.push_str(&format!("    if std::env::var_os(\"CARGO_FEATURE_{}\").is_some() {{ build_{n}::run(); }}\n", n.to_uppercase()));
    }
    s.push_str("}\n");
    Some(s)
}

/// The `[dependencies]` entries of a backend manifest as `(crate name, normalized line)`.
fn dependency_lines(manifest: &str) -> Vec<(String, String)> {
    let mut in_deps = false;
    let mut out = Vec::new();
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_deps = trimmed == "[dependencies]";
            continue;
        }
        if !in_deps || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let name = trimmed.split('=').next().unwrap_or("").trim().to_string();
        let normalized = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
        out.push((name, normalized));
    }
    out
}

/// The combined manifest, plus `(dependency key, crate identifier)` pairs per backend for the
/// crates that had to be renamed because backends need different versions of them.
struct Manifest {
    text: String,
    aliases: BTreeMap<Backend, Vec<(String, String)>>,
}

/// Split `name = "1"` / `name = { version = "1", ... }` into the crate name and the table body.
fn split_dependency(line: &str) -> (String, String) {
    let (name, rest) = line.split_once('=').unwrap_or((line, ""));
    let rest = rest.trim();
    let inner = match (rest.strip_prefix('{'), rest.rfind('}')) {
        (Some(_), Some(close)) => rest[1..close].trim().to_string(),
        _ => format!("version = {rest}"),
    };
    (name.trim().to_string(), inner)
}

fn render_dependency(key: &str, package: Option<&str>, inner: &str, optional: bool) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(package) = package {
        parts.push(format!("package = \"{package}\""));
    }
    parts.push(inner.to_string());
    if optional {
        parts.push("optional = true".to_string());
    }
    format!("{key} = {{ {} }}", parts.join(", "))
}

fn combined_manifest(stem: &str, version: &str, units: &[Unit]) -> Manifest {
    // crate name -> the distinct specs requested, each with the backends asking for it.
    let mut variants: BTreeMap<String, Vec<(String, Vec<Backend>)>> = BTreeMap::new();
    for unit in units {
        for (name, line) in dependency_lines(&unit.cargo_toml) {
            let specs = variants.entry(name).or_default();
            match specs.iter_mut().find(|(existing, _)| *existing == line) {
                Some((_, users)) => users.push(unit.backend),
                None => specs.push((line, vec![unit.backend])),
            }
        }
    }

    // One `[dependencies]` entry per spec. Backends that disagree on a crate (e.g. winit 0.28 vs
    // 0.30) each get it under a distinct key; their module imports it back under its real name.
    struct Entry {
        key: String,
        line: String,
        users: Vec<Backend>,
    }
    let mut entries: Vec<Entry> = Vec::new();
    let mut aliases: BTreeMap<Backend, Vec<(String, String)>> = BTreeMap::new();
    for (name, specs) in &variants {
        for (line, users) in specs {
            if specs.len() == 1 {
                entries.push(Entry { key: name.clone(), line: line.clone(), users: users.clone() });
            } else {
                let suffix: Vec<&str> = users.iter().map(|b| b.name()).collect();
                let key = format!("{}_{}", name.replace('-', "_"), suffix.join("_"));
                let (_, inner) = split_dependency(line);
                entries.push(Entry { key: key.clone(), line: render_dependency(&key, Some(name), &inner, false), users: users.clone() });
                for backend in users {
                    aliases.entry(*backend).or_default().push((key.clone(), name.replace('-', "_")));
                }
            }
        }
    }

    let backend_names: Vec<&str> = units.iter().map(|u| u.backend.name()).collect();
    let mut m = String::new();
    m.push_str(&format!("[package]\nname = \"{stem}\"\nversion = \"{version}\"\nedition = \"2024\"\n\n"));
    m.push_str(&format!("[[bin]]\nname = \"{stem}\"\npath = \"src/main.rs\"\n\n"));
    m.push_str("[profile.release]\nopt-level = 3\nlto = \"thin\"\ncodegen-units = 1\n\n");

    m.push_str("[features]\n");
    m.push_str(&format!("default = [{}]\n", backend_names.iter().map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(", ")));
    for unit in units {
        // A crate only one backend needs is optional and pulled in by that backend's feature;
        // a crate several backends share is always built (small, and avoids feature algebra).
        let own: Vec<String> = entries
            .iter()
            .filter(|e| e.users.as_slice() == [unit.backend])
            .map(|e| format!("\"dep:{}\"", e.key))
            .collect();
        m.push_str(&format!("{} = [{}]\n", unit.backend.name(), own.join(", ")));
    }

    m.push_str("\n[dependencies]\n");
    for entry in &entries {
        if entry.users.len() == 1 {
            let (_, inner) = split_dependency(&entry.line);
            // An already-renamed entry keeps its `package = ...`, which sits inside `inner`.
            m.push_str(&render_dependency(&entry.key, None, &inner, true));
        } else {
            m.push_str(&entry.line);
        }
        m.push('\n');
    }
    Manifest { text: m, aliases }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lists_and_all() {
        let (b, _) = parse_targets("metal,cuda,metal").unwrap();
        assert_eq!(b, vec![Backend::Cuda, Backend::Metal]);
        assert!(parse_targets("cuda,bogus").is_err());
        let (all, skipped) = parse_targets("all").unwrap();
        assert!(all.contains(&Backend::Wgpu) && all.contains(&Backend::Simd));
        assert_eq!(all.len() + skipped.len(), 5);
    }

    #[test]
    fn module_rename_and_probe() {
        let src = "#![allow(dead_code)]\n#[macro_use] extern crate objc;\nfn main() {\n}\n";
        let m = module_source(Backend::Simd, src, &[]).unwrap();
        assert!(!m.contains("extern crate"));
        assert!(m.contains("pub fn __boring_backend_main("));
        assert!(m.contains("__boring_entry"));
        assert!(module_source(Backend::Simd, "fn other() {}\n", &[]).is_err());
    }

    #[test]
    fn manifest_makes_single_backend_deps_optional() {
        let unit = |backend, toml: &str| Unit { backend, host_rs: String::new(), files: vec![], cargo_toml: toml.into(), build_rs: None, routes: vec![] };
        let units = [
            unit(Backend::Metal, "[dependencies]\nmetal = \"0.29\"\nhalf = { version = \"2\", features = [\"bytemuck\"] }\n"),
            unit(Backend::Wgpu, "[dependencies]\nwgpu = \"22\"\nhalf = { version = \"2\", features = [\"bytemuck\"] }\n"),
        ];
        let m = combined_manifest("app", "0.1.0", &units).text;
        assert!(m.contains("metal = { version = \"0.29\", optional = true }"), "{m}");
        assert!(m.contains("metal = [\"dep:metal\"]"), "{m}");
        assert!(m.contains("half = { version = \"2\", features = [\"bytemuck\"] }\n"), "{m}");
    }

    #[test]
    fn conflicting_versions_get_distinct_keys_and_aliases() {
        let unit = |backend, toml: &str| Unit { backend, host_rs: String::new(), files: vec![], cargo_toml: toml.into(), build_rs: None, routes: vec![] };
        let units = [
            unit(Backend::Metal, "[dependencies]\nwinit = \"0.28\"\n"),
            unit(Backend::Simd, "[dependencies]\nwinit = \"0.28\"\n"),
            unit(Backend::Wgpu, "[dependencies]\nwinit = { version = \"0.30\", features = [\"x\"] }\n"),
        ];
        let m = combined_manifest("app", "0.1.0", &units);
        assert!(m.text.contains("winit_metal_simd = { package = \"winit\", version = \"0.28\" }"), "{}", m.text);
        assert!(m.text.contains("winit_wgpu = { package = \"winit\", version = \"0.30\", features = [\"x\"], optional = true }"), "{}", m.text);
        assert!(m.text.contains("wgpu = [\"dep:winit_wgpu\"]"), "{}", m.text);
        assert_eq!(m.aliases[&Backend::Wgpu], vec![("winit_wgpu".to_string(), "winit".to_string())]);
        assert_eq!(m.aliases[&Backend::Simd][0].0, "winit_metal_simd");
    }
}
