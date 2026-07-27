// Copyright (C) 2026 MickaÃ«l LANOÃ‹
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of Boring.
// Boring is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// See the LICENSE file at the project root for the full text.

// Signal is used for interpreter control flow; boxing Value variants would add heap allocations in the hot loop
#![allow(clippy::result_large_err)]

pub mod errors;
pub mod lexer;
pub mod ast;
pub mod parser;
pub mod interpreter;
pub mod checker;
pub mod transpiler;
pub mod validator;

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process;

// â”€â”€â”€ Diagnostics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

struct Ansi { on: bool }
impl Ansi {
    fn new() -> Self { Self { on: std::io::stderr().is_terminal() } }
    fn red(&self, s: &str)    -> String { if self.on { format!("\x1b[31;1m{s}\x1b[0m") } else { s.to_string() } }
    fn blue(&self, s: &str)   -> String { if self.on { format!("\x1b[34;1m{s}\x1b[0m") } else { s.to_string() } }
    fn dim(&self, s: &str)    -> String { if self.on { format!("\x1b[2m{s}\x1b[0m")    } else { s.to_string() } }
    fn bold(&self, s: &str)   -> String { if self.on { format!("\x1b[1m{s}\x1b[0m")    } else { s.to_string() } }
    fn yellow(&self, s: &str) -> String { if self.on { format!("\x1b[33;1m{s}\x1b[0m") } else { s.to_string() } }
}

/// Print a Rust-style diagnostic with source context.
///
/// ```text
/// error: <message>
///  --> path/to/file.br:5:3
///   |
/// 5 | let z = x + y
///   |   ^
/// ```
fn report_error(path: &Path, source: &str, line: usize, col: usize, len: usize, message: &str) {
    let c = Ansi::new();
    eprintln!("{} {}", c.red("error:"), c.bold(message));
    if line == 0 {
        eprintln!(" {} {}", c.blue("-->"), path.display());
        return;
    }
    let col_s = if col > 0 { format!(":{}", col) } else { String::new() };
    eprintln!(" {} {}:{}{}", c.blue("-->"), path.display(), line, col_s);
    let width = line.to_string().len();
    let pad   = " ".repeat(width);
    eprintln!("{} {}", pad, c.dim("|"));
    let src_line = source.lines().nth(line.saturating_sub(1)).unwrap_or("");
    eprintln!("{} {} {}", c.dim(&line.to_string()), c.dim("|"), src_line);
    if col > 0 {
        let underline = "^".repeat(len.max(1));
        eprintln!("{} {} {}", pad, c.dim("|"), c.red(&format!("{}{}", " ".repeat(col.saturating_sub(1)), underline)));
    } else {
        eprintln!("{} {}", pad, c.dim("|"));
    }
}

fn report_warning(path: &Path, source: &str, line: usize, col: usize, len: usize, message: &str) {
    let c = Ansi::new();
    eprintln!("{} {}", c.yellow("warning:"), c.bold(message));
    if line == 0 {
        eprintln!(" {} {}", c.blue("-->"), path.display());
        return;
    }
    let col_s = if col > 0 { format!(":{}", col) } else { String::new() };
    eprintln!(" {} {}:{}{}", c.blue("-->"), path.display(), line, col_s);
    let width = line.to_string().len();
    let pad   = " ".repeat(width);
    eprintln!("{} {}", pad, c.dim("|"));
    let src_line = source.lines().nth(line.saturating_sub(1)).unwrap_or("");
    eprintln!("{} {} {}", c.dim(&line.to_string()), c.dim("|"), src_line);
    if col > 0 {
        let underline = "^".repeat(len.max(1));
        eprintln!("{} {} {}", pad, c.dim("|"), c.yellow(&format!("{}{}", " ".repeat(col.saturating_sub(1)), underline)));
    } else {
        eprintln!("{} {}", pad, c.dim("|"));
    }
}

fn report_lex_errors(path: &Path, source: &str, errors: &[lexer::LexError]) {
    for e in errors {
        report_error(path, source, e.line(), e.col(), e.len(), &e.msg());
    }
}

fn report_transpile_errors(path: &Path, source: &str, errors: &[transpiler::TranspileError]) {
    for e in errors {
        report_error(path, source, e.line, e.col, e.len, &e.message);
    }
}

fn report_transpile_warnings(path: &Path, source: &str, warnings: &[transpiler::TranspileError]) {
    for w in warnings {
        report_warning(path, source, w.line, w.col, w.len, &w.message);
    }
}

fn report_check_result(path: &Path, source: &str, result: checker::CheckResult) -> bool {
    for w in &result.warnings {
        report_warning(path, source, w.line, w.col, w.len, &w.message);
    }
    for e in &result.errors {
        report_error(path, source, e.line, e.col, e.len, &e.message);
    }
    !result.errors.is_empty()
}

// â”€â”€â”€ boring.toml â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Minimal `boring.toml` representation.
struct BoringToml {
    name:    String,
    version: String,
    main:    String,
}

impl BoringToml {
    /// Parse a `boring.toml` file.  No external dependency â€” the format is tiny.
    fn parse(src: &str) -> Self {
        let mut name    = String::new();
        let mut version = "0.1.0".to_string();
        let mut main    = "main.br".to_string();

        for line in src.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("name") {
                if let Some(v) = Self::extract_value(rest) { name = v; }
            } else if let Some(rest) = line.strip_prefix("version") {
                if let Some(v) = Self::extract_value(rest) { version = v; }
            } else if let Some(rest) = line.strip_prefix("main") {
                if let Some(v) = Self::extract_value(rest) { main = v; }
            }
        }
        BoringToml { name, version, main }
    }

    /// Extract the value from `= "value"` or `= value`.
    fn extract_value(rest: &str) -> Option<String> {
        let rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == '=').trim();
        if rest.starts_with('"') {
            Some(rest.trim_matches('"').to_string())
        } else if !rest.is_empty() {
            Some(rest.to_string())
        } else {
            None
        }
    }
}

/// Load `boring.toml` from the current directory, or exit with an error.
fn load_project_toml() -> (BoringToml, PathBuf) {
    let toml_path = PathBuf::from("boring.toml");
    let src = match std::fs::read_to_string(&toml_path) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("error: no boring.toml found in current directory");
            eprintln!("hint:  use `boring new <name>` to create a new project");
            process::exit(1);
        }
    };
    let toml = BoringToml::parse(&src);
    if toml.name.is_empty() {
        eprintln!("error: boring.toml is missing a `name` field");
        process::exit(1);
    }
    (toml, toml_path)
}

/// Walk up from `start` (a file or directory) looking for an ancestor directory
/// containing `boring.toml`, treating it as the project root. Returns `None` if
/// no such ancestor exists (e.g. a standalone script run outside any project).
///
/// Used by `boring run` so a script under `test/` or `examples/` can `use` a
/// module that lives under the project's `src/` directory without requiring
/// `BORING_PATH` to be set manually — mirroring the `src/`-rooted layout
/// `boring build` already assumes for a project's `boring.toml`.
fn find_project_root(start: &Path) -> Option<PathBuf> {
    // Resolve to an absolute path first: a relative `start` like "foo.br" (no
    // directory component) must still walk up from the real current directory,
    // not stop after checking CWD once because the relative path ran out of
    // components.
    let absolute = start.canonicalize().ok()?;
    let mut dir: &Path = if absolute.is_dir() { &absolute } else { absolute.parent()? };
    loop {
        if dir.join("boring.toml").is_file() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
}

// â”€â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn main() {
    // Windows default stack is 1 MB â€” too small for the interpreter's recursive
    // descent (expression evaluation, type resolution, pattern matching).
    // Spawn a new thread with a much larger stack so the same code runs
    // everywhere. Each level of *interpreted* recursion (e.g. a user-defined
    // function calling itself) costs far more native stack than it looks like
    // it should: one Boring-level call spans a whole chain of native frames
    // (eval_expr -> call_fn -> exec_block -> exec_stmt -> eval_expr -> ...),
    // and in an unoptimized debug build of `boring` itself, the interpreter's
    // large match-based eval_expr/exec_stmt functions don't get their stack
    // slots reused across match arms the way a release build's optimizer
    // would, multiplying the per-call cost roughly 10x. With the previous
    // 8 MB, a debug build of the interpreter overflowed on a plain recursive
    // function (e.g. a factorial) after only ~15-20 levels of recursion —
    // even a release build only reached ~190 levels. 256 MB is only a
    // virtual-memory reservation (pages are committed lazily as the stack
    // actually grows), so it costs nothing for shallow programs; it raises
    // that ceiling to roughly the thousands in debug builds and much higher
    // in release builds.
    const STACK_SIZE: usize = 256 * 1024 * 1024; // 256 MB
    let builder = std::thread::Builder::new().stack_size(STACK_SIZE);
    let handler = builder.spawn(run).expect("failed to spawn main thread");
    match handler.join() {
        Ok(()) => {}
        Err(e) => {
            // The worker thread panicked â€” print the payload for diagnosis.
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                format!("internal error: {}", s)
            } else if let Some(s) = e.downcast_ref::<String>() {
                format!("internal error: {}", s)
            } else {
                "internal error: unexpected panic in worker thread".to_string()
            };
            eprintln!("{}", msg);
            std::process::exit(101);
        }
    }
}

/// Parse `boring run [--gpu <profile>] [file.br]` flags.
/// Returns (gpu_profile_name, file_path).
fn parse_run_flags(args: &[String]) -> (Option<String>, Option<&str>) {
    let mut gpu: Option<String> = None;
    let mut file: Option<&str>  = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gpu" => {
                i += 1;
                if let Some(name) = args.get(i) {
                    gpu = Some(name.clone());
                } else {
                    eprintln!("error: --gpu requires a profile name");
                    process::exit(1);
                }
            }
            "--" => {
                // Everything after `--` is passed to the script via args() — stop parsing here.
                break;
            }
            s if !s.starts_with('-') => {
                if file.is_some() {
                    eprintln!("error: unexpected extra argument '{s}' (a file was already given)");
                    process::exit(1);
                }
                file = Some(args[i].as_str());
            }
            other => {
                eprintln!("error: unknown run flag '{other}'");
                process::exit(1);
            }
        }
        i += 1;
    }
    (gpu, file)
}

fn run() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("--help") | Some("-h") => {
            print_help();
            process::exit(0);
        }

        // â”€â”€ Project commands â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        Some("new") => {
            let name = args.get(2).unwrap_or_else(|| {
                eprintln!("usage: boring new <name>");
                process::exit(1);
            });
            new_project(name);
        }
        Some("run") => {
            // Parse optional: --gpu <profile>  before the file path.
            let run_args = &args[2..];
            let (gpu_profile_name, file_arg) = parse_run_flags(run_args);
            match file_arg {
                Some(path) => run_file(path, gpu_profile_name.as_deref()),
                None       => run_project(),
            }
        }
        Some("build") => {
            let build_args = &args[2..];
            parse_build_command(build_args);
        }

        Some(path) => run_file(path, None),
        None => {
            print_help();
            process::exit(1);
        }
    }
}

fn print_help() {
    eprintln!("Boring language runner and transpiler");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    boring new <name>          Create a new project");
    eprintln!("    boring run                 Run the project in the current directory");
    eprintln!("    boring run <file.br>       Run a single file");
    eprintln!("    boring run --gpu <profile> <file.br>  Run with a GPU simulation profile");
    eprintln!("                               Built-in profiles: default, v100, a100, rtx3090, rtx4090, h100");
    eprintln!("                               Custom profile:    --gpu path/to/profile.toml");
    eprintln!("    boring build               Emit a Cargo project from boring.toml");
    eprintln!("    boring build <file.br>     Emit a Cargo project from a single file");
    eprintln!("    boring build --mode managed              Use managed memory mode (Arc<Mutex> defaults)");
    eprintln!("    boring build --threading single          Use single-thread Tokio runtime");
    eprintln!("    boring build --instrument                Instrument functions (coverage + trace journals)
    boring build --emit-rust                 Print generated Rust source to stdout (no project created)
    boring build --sanitize address|thread|memory  Enable a sanitizer (requires nightly toolchain)
    boring build --compile                   Transpile then run cargo build in the generated project
    boring build --rust-options \"<flags>\"   Pass extra flags to cargo build (implies --compile)
                                            Example: --rust-options \"--release\"");
    eprintln!("    boring build --target kernel             Emit a kernel Cargo project from boring.toml");
    eprintln!("    boring build --target kernel <file.br>   Emit a kernel Cargo project from a single file");
    eprintln!("    boring build --target cuda               Emit a CUDA Cargo project from boring.toml");
    eprintln!("    boring build --target cuda <file.br>     Emit a CUDA Cargo project from a single file");
    eprintln!("    boring build --target metal              Emit a Metal Cargo project from boring.toml");
    eprintln!("    boring build --target metal <file.br>    Emit a Metal Cargo project from a single file");
    eprintln!("    boring build --target wgpu               Emit a wgpu Cargo project from boring.toml (Windows/Linux/macOS)");
    eprintln!("    boring build --target wgpu <file.br>     Emit a wgpu Cargo project from a single file");
    eprintln!("    boring <file.br>           Run a single file (shorthand)");
}

// â”€â”€â”€ Project commands â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// `boring new <name>` â€” scaffold a project directory.
fn new_project(name: &str) {
    // Validate name
    if name.starts_with('-') || name.contains('/') || name.contains('\\') {
        eprintln!("error: invalid project name '{}'", name);
        process::exit(1);
    }

    let project_dir = PathBuf::from(name);
    if project_dir.exists() {
        eprintln!("error: directory '{}' already exists", name);
        process::exit(1);
    }

    // Create directory
    if let Err(e) = std::fs::create_dir(&project_dir) {
        eprintln!("error: cannot create directory '{}': {}", name, e);
        process::exit(1);
    }

    // Write boring.toml
    let toml_content = format!(
        "[project]\nname = \"{name}\"\nversion = \"0.1.0\"\n"
    );
    let toml_path = project_dir.join("boring.toml");
    if let Err(e) = std::fs::write(&toml_path, toml_content) {
        eprintln!("error: cannot write boring.toml: {}", e);
        process::exit(1);
    }

    // Write main.br
    let main_content = format!("print \"Hello from {name}!\"\n");
    let main_path = project_dir.join("main.br");
    if let Err(e) = std::fs::write(&main_path, main_content) {
        eprintln!("error: cannot write main.br: {}", e);
        process::exit(1);
    }

    eprintln!("Created project '{}'", name);
    eprintln!("    cd {name} && boring run");
}

/// `boring run` â€” run the project described by `boring.toml` in the current directory.
fn run_project() {
    let (toml, _) = load_project_toml();
    run_file(&toml.main, None);
}

/// `boring build` — emit a Cargo project from the `boring.toml` main file.
fn build_project_with_config(config: transpiler::TranspileConfig) {
    let (toml, _) = load_project_toml();
    emit_rust_with_version_and_config(&toml.main, &toml.version, config);
}

/// `boring build --target kernel` â€” emit a kernel Cargo project from `boring.toml`.
fn build_project_kernel() {
    let (toml, _) = load_project_toml();
    emit_kernel_with_version(&toml.main, &toml.version);
}

/// Parse `boring build [flags] [file.br]` arguments after the `build` subcommand.
fn parse_build_command(build_args: &[String]) {
    use transpiler::{TranspileConfig, TranspileMode, ThreadingMode};

    let mut target_kernel = false;
    let mut target_cuda   = false;
    let mut target_metal  = false;
    let mut target_wgpu   = false;
    let mut mode = TranspileMode::Strict;
    let mut threading = ThreadingMode::Multi;
    let mut stack_auto_bytes: usize = 256;
    let mut instrument = false;
    let mut sanitize: Option<&'static str> = None;
    let mut emit_rust = false;
    let mut compile = false;
    let mut rust_options: Vec<String> = Vec::new();
    let mut file: Option<&str> = None;
    let mut output_dir: Option<PathBuf> = None;

    let mut i = 0;
    while i < build_args.len() {
        match build_args[i].as_str() {
            "--target" => {
                i += 1;
                match build_args.get(i).map(|s| s.as_str()) {
                    Some("kernel") => target_kernel = true,
                    Some("cuda")   => target_cuda   = true,
                    Some("metal")  => target_metal  = true,
                    Some("wgpu")   => target_wgpu   = true,
                    Some(t) => {
                        eprintln!("error: unknown target '{}'", t);
                        eprintln!("hint:  supported targets: kernel, cuda, metal, wgpu");
                        process::exit(1);
                    }
                    None => {
                        eprintln!("error: --target requires a value");
                        eprintln!("hint:  supported targets: kernel, cuda, metal, wgpu");
                        process::exit(1);
                    }
                }
            }
            "--mode" => {
                i += 1;
                match build_args.get(i).map(|s| s.as_str()) {
                    Some("strict")  => mode = TranspileMode::Strict,
                    Some("managed") => mode = TranspileMode::Managed,
                    Some(m) => {
                        eprintln!("error: unknown mode '{}' â€” expected strict or managed", m);
                        process::exit(1);
                    }
                    None => {
                        eprintln!("error: --mode requires a value (strict or managed)");
                        process::exit(1);
                    }
                }
            }
            "--threading" => {
                i += 1;
                match build_args.get(i).map(|s| s.as_str()) {
                    Some("multi")  => threading = ThreadingMode::Multi,
                    Some("single") => threading = ThreadingMode::Single,
                    Some(t) => {
                        eprintln!("error: unknown threading model '{}' â€” expected single or multi", t);
                        process::exit(1);
                    }
                    None => {
                        eprintln!("error: --threading requires a value (single or multi)");
                        process::exit(1);
                    }
                }
            }
            "--output-dir" => {
                i += 1;
                match build_args.get(i) {
                    Some(dir) => output_dir = Some(PathBuf::from(dir)),
                    None => {
                        eprintln!("error: --output-dir requires a path");
                        process::exit(1);
                    }
                }
            }
            "--stack-auto-bytes" => {
                i += 1;
                match build_args.get(i).and_then(|s| s.parse::<usize>().ok()) {
                    Some(n) => stack_auto_bytes = n,
                    None => {
                        eprintln!("error: --stack-auto-bytes requires a positive integer");
                        process::exit(1);
                    }
                }
            }
            "--instrument" => instrument = true,
            "--emit-rust"  => emit_rust = true,
            "--compile"    => compile = true,
            "--rust-options" => {
                i += 1;
                match build_args.get(i) {
                    Some(opts) => {
                        rust_options.extend(opts.split_whitespace().map(|s| s.to_string()));
                        compile = true;
                    }
                    None => {
                        eprintln!("error: --rust-options requires a value");
                        process::exit(1);
                    }
                }
            }
            "--sanitize" => {
                i += 1;
                match build_args.get(i).map(|s| s.as_str()) {
                    Some("address") => sanitize = Some("address"),
                    Some("thread")  => sanitize = Some("thread"),
                    Some("memory")  => sanitize = Some("memory"),
                    Some(s) => {
                        eprintln!("error: unknown sanitizer '{}' â€” expected address, thread, or memory", s);
                        process::exit(1);
                    }
                    None => {
                        eprintln!("error: --sanitize requires a value (address, thread, or memory)");
                        process::exit(1);
                    }
                }
            }
            arg if !arg.starts_with('-') => {
                file = Some(&build_args[i]);
            }
            arg => {
                eprintln!("error: unknown flag '{}'", arg);
                process::exit(1);
            }
        }
        i += 1;
    }

    if target_kernel && threading != ThreadingMode::Multi {
        eprintln!("error: --threading is not available for the kernel target");
        process::exit(1);
    }

    if target_cuda && threading != ThreadingMode::Multi {
        eprintln!("error: --threading is not available for the cuda target");
        process::exit(1);
    }
    if target_metal && threading != ThreadingMode::Multi {
        eprintln!("error: --threading is not available for the metal target");
        process::exit(1);
    }

    if target_wgpu && threading != ThreadingMode::Multi {
        eprintln!("error: --threading is not available for the wgpu target");
        process::exit(1);
    }

    // CUDA target â€” short-circuit before the general config path.
    if target_cuda {
        match file {
            Some(path) => { emit_cuda(path, "0.1.0"); return; }
            None => {
                let (toml, _) = load_project_toml();
                emit_cuda(&toml.main, &toml.version);
                return;
            }
        }
    }

    // Metal target.
    if target_metal {
        match file {
            Some(path) => { emit_metal(path, "0.1.0"); return; }
            None => {
                let (toml, _) = load_project_toml();
                emit_metal(&toml.main, &toml.version);
                return;
            }
        }
    }

    // wgpu target.
    if target_wgpu {
        match file {
            Some(path) => { emit_wgpu(path, "0.1.0"); return; }
            None => {
                let (toml, _) = load_project_toml();
                emit_wgpu(&toml.main, &toml.version);
                return;
            }
        }
    }

    let config = TranspileConfig { mode, threading, stack_auto_bytes, instrument, sanitize, source_dir: PathBuf::new(), gpu_kernels: Vec::new(), is_gpu_target: false, gpu_top_level_handled_by_host: false };

    if emit_rust {
        match file {
            Some(path) => print_rust(path, config),
            None => {
                let (toml, _) = load_project_toml();
                print_rust(&toml.main, config);
            }
        }
        return;
    }

    let project_dir = match (target_kernel, file, output_dir) {
        (true,  Some(path), _)          => { emit_kernel(path); return; }
        (true,  None, _)                => { build_project_kernel(); return; }
        (false, Some(path), Some(dir))  => { emit_rust_to_dir(path, "0.1.0", config, dir.clone()); dir }
        (false, Some(path), None)       => {
            let stem = std::path::Path::new(path)
                .file_stem().map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "output".to_string());
            let base = std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."));
            let dir = base.join(rust_dir_name_full(&stem, &config.threading, &config.mode));
            emit_rust_with_config(path, config);
            dir
        }
        (false, None, _)                => {
            let (toml, _) = load_project_toml();
            let stem = std::path::Path::new(&toml.main)
                .file_stem().map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "output".to_string());
            let base = std::path::Path::new(&toml.main).parent().unwrap_or(std::path::Path::new("."));
            let dir = base.join(rust_dir_name_full(&stem, &config.threading, &config.mode));
            build_project_with_config(config);
            dir
        }
    };

    if compile {
        run_cargo_build(&project_dir, &rust_options);
    }
}

fn run_cargo_build(project_dir: &PathBuf, rust_options: &[String]) {
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("build");
    cmd.args(rust_options);
    cmd.current_dir(project_dir);

    eprintln!("Running: cargo build {}", rust_options.join(" "));
    match cmd.status() {
        Ok(status) if status.success() => {}
        Ok(status) => {
            eprintln!("error: cargo build exited with status {}", status);
            process::exit(status.code().unwrap_or(1));
        }
        Err(e) => {
            eprintln!("error: failed to invoke cargo: {}", e);
            process::exit(1);
        }
    }
}

// â”€â”€â”€ Core: interpret â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn run_file(path: &str, gpu_profile: Option<&str>) {
    let path = PathBuf::from(path);

    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read '{}': {}", path.display(), e);
            process::exit(1);
        }
    };

    let tokens = match lexer::lex_all(&source) {
        Ok(t) => t,
        Err(errors) => {
            report_lex_errors(&path, &source, &errors);
            process::exit(1);
        }
    };

    let program = match parser::parse(tokens) {
        Ok(p) => p,
        Err(e) => {
            report_error(&path, &source, e.line(), e.col(), e.len(), &e.msg());
            process::exit(1);
        }
    };

    if report_check_result(&path, &source, checker::check(&program)) {
        process::exit(1);
    }

    let mut interp = interpreter::Interpreter::new();

    if let Some(name) = gpu_profile {
        match interpreter::gpu_profile::GpuProfile::load(name) {
            Ok(profile) => interp.gpu_profile = profile,
            Err(e) => {
                eprintln!("error: {e}");
                eprintln!("available profiles: default, v100, a100, rtx3090, rtx4090, h100");
                process::exit(1);
            }
        }
    }

    // Add the file's directory to the search path for `use` resolution
    if let Some(dir) = path.parent() {
        interp.add_search_path(dir.to_path_buf());
    }

    // If this file belongs to a project (an ancestor directory has a
    // `boring.toml`), also search that project's `src/` directory — so a
    // script under `test/` or `examples/` can `use` a module that lives in
    // `src/` without needing `BORING_PATH` set. See `find_project_root`.
    if let Some(root) = find_project_root(&path) {
        let src_dir = root.join("src");
        if src_dir.is_dir() {
            interp.add_search_path(src_dir);
        }
    }

    // Add BORING_PATH entries (uses OS path-list separator: `:` on Unix, `;` on Windows)
    if let Ok(env_path) = std::env::var("BORING_PATH") {
        for p in std::env::split_paths(&env_path) {
            interp.add_search_path(p);
        }
    }

    if let Err(e) = interp.exec_program(&program) {
        report_error(&path, &source, e.line, e.col, e.len, &e.message);
        process::exit(1);
    }

    // After execution: write PPM for every Screen that received at least one frame.
    // In simulation mode without --preview, this is the only visual output.
    let screens: Vec<(String, interpreter::Value)> = {
        let g = interp.global.borrow();
        g.all_bindings().into_iter()
            .filter(|(_, v)| matches!(v, interpreter::Value::Screen { frame, .. } if *frame.borrow() > 0))
            .collect()
    };
    for (name, val) in screens {
        if let interpreter::Value::Screen { pixels, width, height, .. } = val {
            let w = *width.borrow() as usize;
            let h = *height.borrow() as usize;
            let px = pixels.borrow();
            let stem = path.file_stem().unwrap_or_default().to_string_lossy();
            let ppm_path = path.with_file_name(format!("{}_{}.ppm", stem, name));
            if let Ok(mut f) = std::fs::File::create(&ppm_path) {
                use std::io::Write;
                let _ = writeln!(f, "P6\n{} {}\n255", w, h);
                for pixel in px.iter().take(w * h) {
                    let r = ((*pixel >> 16) & 0xFF) as u8;
                    let g = ((*pixel >>  8) & 0xFF) as u8;
                    let b = ( *pixel        & 0xFF) as u8;
                    let _ = f.write_all(&[r, g, b]);
                }
                eprintln!("wrote {}", ppm_path.display());
            }
        }
    }
}

// â”€â”€â”€ Core: transpile â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn emit_rust_with_config(path: &str, config: transpiler::TranspileConfig) {
    emit_rust_with_version_and_config(path, "0.1.0", config);
}

/// Transpile a `.br` file and print the generated Rust source to stdout.
fn print_rust(path: &str, config: transpiler::TranspileConfig) {
    let path = PathBuf::from(path);
    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read '{}': {}", path.display(), e);
            process::exit(1);
        }
    };
    let tokens = match lexer::lex_all(&source) {
        Ok(t) => t,
        Err(errors) => { report_lex_errors(&path, &source, &errors); process::exit(1); }
    };
    let program = match parser::parse(tokens) {
        Ok(p) => p,
        Err(e) => { report_error(&path, &source, e.line(), e.col(), e.len(), &e.msg()); process::exit(1); }
    };
    if report_check_result(&path, &source, checker::check(&program)) { process::exit(1); }
    let out = transpiler::transpile_with_config(&program, config);
    report_transpile_warnings(&path, &source, &out.warnings);
    if !out.errors.is_empty() { report_transpile_errors(&path, &source, &out.errors); process::exit(1); }
    print!("{}", out.code);
}

fn rust_dir_name_full(stem: &str, threading: &transpiler::ThreadingMode, mode: &transpiler::TranspileMode) -> String {
    let managed = matches!(mode, transpiler::TranspileMode::Managed);
    let single  = matches!(threading, transpiler::ThreadingMode::Single);
    match (managed, single) {
        (false, false) => format!("{}_rust", stem),
        (false, true)  => format!("{}_rust_single", stem),
        (true,  false) => format!("{}_rust_managed", stem),
        (true,  true)  => format!("{}_rust_managed_single", stem),
    }
}

fn emit_rust_with_version_and_config(path: &str, version: &str, config: transpiler::TranspileConfig) {
    let path = PathBuf::from(path);

    // Determine output project directory next to the source file
    let stem = path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let base_dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let project_dir = base_dir.join(rust_dir_name_full(&stem, &config.threading, &config.mode));

    emit_rust_to_dir(path.to_str().unwrap_or(""), version, config, project_dir);
}

fn emit_rust_to_dir(path: &str, version: &str, config: transpiler::TranspileConfig, project_dir: PathBuf) {
    let path = PathBuf::from(path);

    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read '{}': {}", path.display(), e);
            process::exit(1);
        }
    };

    let tokens = match lexer::lex_all(&source) {
        Ok(t) => t,
        Err(errors) => {
            report_lex_errors(&path, &source, &errors);
            process::exit(1);
        }
    };

    let program = match parser::parse(tokens) {
        Ok(p) => p,
        Err(e) => {
            report_error(&path, &source, e.line(), e.col(), e.len(), &e.msg());
            process::exit(1);
        }
    };

    if report_check_result(&path, &source, checker::check(&program)) {
        process::exit(1);
    }

    let source_dir = path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
    let config_with_dir = transpiler::TranspileConfig { source_dir, ..config.clone() };
    let transpile_out = transpiler::transpile_with_config(&program, config_with_dir);
    report_transpile_warnings(&path, &source, &transpile_out.warnings);
    if !transpile_out.errors.is_empty() { report_transpile_errors(&path, &source, &transpile_out.errors); process::exit(1); }
    let rust_code = transpile_out.code;
    let has_streams = transpile_out.has_streams;
    let uses_log = transpile_out.uses_log;
    let uses_thiserror = transpile_out.uses_thiserror;
    let uses_reqwest = transpile_out.uses_reqwest;
    let uses_tokio_util = transpile_out.uses_tokio_util;
    let uses_serde = transpile_out.uses_serde;
    let uses_local_channel = transpile_out.uses_local_channel;
    let extra_modules = transpile_out.modules;

    let stem = path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());

    // Create directory layout
    let src_dir = project_dir.join("src");
    if let Err(e) = std::fs::create_dir_all(&src_dir) {
        eprintln!("error: cannot create '{}': {}", src_dir.display(), e);
        process::exit(1);
    }

    // Write per-module .rs files (one per `use <file>.br` import).
    // Each file is included into main.rs via `include!` so they share the same flat namespace.
    let mut include_prefix = String::new();
    for (mod_name, mod_code) in &extra_modules {
        let mod_path = src_dir.join(format!("{}.rs", mod_name));
        if let Err(e) = std::fs::write(&mod_path, mod_code) {
            eprintln!("error: cannot write '{}': {}", mod_path.display(), e);
            process::exit(1);
        }
        include_prefix.push_str(&format!("include!(\"{}.rs\");\n", mod_name));
    }

    // Write src/main.rs  (strip the old standalone rustc comment if present)
    let clean_code = rust_code
        .strip_prefix("// Generated by boring build. Compile with: rustc --edition 2021 <file.rs>\n")
        .unwrap_or(&rust_code);
    let main_rs_code = if include_prefix.is_empty() {
        clean_code.to_string()
    } else {
        format!("{}\n{}", include_prefix, clean_code)
    };
    let main_rs = src_dir.join("main.rs");
    if let Err(e) = std::fs::write(&main_rs, main_rs_code) {
        eprintln!("error: cannot write '{}': {}", main_rs.display(), e);
        process::exit(1);
    }

    // Write Cargo.toml
    let stream_deps = if has_streams {
        "\nasync-stream = \"0.3\"\ntokio-stream = \"0.1\"\nfutures-core = \"0.3\"\n"
    } else {
        ""
    };
    let log_dep = if uses_log {
        "\nlog = \"0.4\"\n"
    } else {
        ""
    };
    let thiserror_dep = if uses_thiserror {
        "\nthiserror = \"1\"\n"
    } else {
        ""
    };
    let reqwest_dep = if uses_reqwest {
        "\nreqwest = { version = \"0.12\", features = [\"json\"] }\n"
    } else {
        ""
    };
    let tokio_util_dep = if uses_tokio_util {
        "\ntokio-util = \"0.7\"\n"
    } else {
        ""
    };
    let serde_dep = if uses_serde {
        "\nserde = { version = \"1\", features = [\"derive\", \"rc\"] }\nserde_json = \"1\"\n"
    } else {
        ""
    };
    let local_channel_dep = if uses_local_channel {
        "\nlocal-channel = \"0.1\"\n"
    } else {
        ""
    };
    let cargo_toml = format!(
        r#"[package]
name = "{stem}"
version = "{version}"
edition = "2024"

[[bin]]
name = "{stem}"
path = "src/main.rs"

[dependencies]
tokio = {{ version = "1", features = ["full"] }}{stream_deps}{log_dep}{thiserror_dep}{reqwest_dep}{tokio_util_dep}{serde_dep}{local_channel_dep}"#
    );
    let cargo_path = project_dir.join("Cargo.toml");
    if let Err(e) = std::fs::write(&cargo_path, cargo_toml) {
        eprintln!("error: cannot write '{}': {}", cargo_path.display(), e);
        process::exit(1);
    }

    // Write .cargo/config.toml when managed mode or --sanitize is requested.
    let needs_cargo_config = config.mode == transpiler::TranspileMode::Managed
        || config.sanitize.is_some();
    if needs_cargo_config {
        let cargo_config_dir = project_dir.join(".cargo");
        if let Err(e) = std::fs::create_dir_all(&cargo_config_dir) {
            eprintln!("error: cannot create '.cargo/': {}", e);
            process::exit(1);
        }
        let mut cargo_config = String::new();
        // Managed mode: automatically enable backtraces so panics show the full call stack.
        if config.mode == transpiler::TranspileMode::Managed {
            cargo_config.push_str("[env]\nRUST_BACKTRACE = \"1\"\n");
        }
        // Sanitizer: inject rustflags targeting the host triple (requires nightly toolchain).
        if let Some(ref san) = config.sanitize {
            let host = host_target();
            if !cargo_config.is_empty() { cargo_config.push('\n'); }
            cargo_config.push_str(&format!(
                "[build]\nrustflags = [\"-Zsanitizer={san}\"]\ntarget = \"{host}\"\n",
                san = san, host = host
            ));
        }
        let config_path = cargo_config_dir.join("config.toml");
        if let Err(e) = std::fs::write(&config_path, cargo_config) {
            eprintln!("error: cannot write '{}': {}", config_path.display(), e);
            process::exit(1);
        }
        if config.sanitize.is_some() {
            eprintln!("note: sanitizer enabled â€” run with: cargo +nightly run");
        }
    }

    eprintln!("Generated Cargo project at '{}'", project_dir.display());
    eprintln!("  cd {} && cargo run", project_dir.display());
}

/// Returns the host Rust target triple by invoking `rustc --version --verbose`.
fn host_target() -> String {
    let output = std::process::Command::new("rustc")
        .args(["--version", "--verbose"])
        .output()
        .ok();
    if let Some(out) = output {
        let s = String::from_utf8_lossy(&out.stdout);
        for line in s.lines() {
            if let Some(target) = line.strip_prefix("host: ") {
                return target.trim().to_string();
            }
        }
    }
    "x86_64-unknown-linux-gnu".to_string()
}

// â”€â”€â”€ Core: kernel transpile â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// â”€â”€â”€ Core: CUDA transpile â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// Parse `path` and recursively inline every `use <file>.br` import reachable
// from it, returning one flattened `Program` with all items concatenated
// (entry file first) and `Item::Use` entries dropped once resolved.
// GPU targets (wgpu/cuda/metal) parse a single file with no `use` resolution
// otherwise, unlike the std/Rust target which resolves `use` imports into
// separate modules via the transpiler's own source_dir-based mechanism.
// A `use` is resolved relative to the importing file's own directory first
// (like the std target), then against each `BORING_PATH` entry (matching
// `boring run`'s search-path behavior — see `run_file`'s `add_search_path`
// calls), so e.g. `test/foo.br` can `use sibling` when `sibling.br` lives in
// `src/` and `BORING_PATH` points there. Circular/duplicate imports are
// visited once. On any read/lex/parse error, reports it in the usual style
// and exits.
fn parse_and_merge_program(path: &str) -> ast::Program {
    let path = PathBuf::from(path);
    let mut visited = std::collections::HashSet::new();
    let mut items = Vec::new();
    let mut search_paths: Vec<PathBuf> = Vec::new();
    // Mirror `run_file`'s project-root discovery: a file under `test/` or
    // `examples/` can `use` a module living in the project's `src/` directory
    // without requiring `BORING_PATH` to be set manually.
    if let Some(root) = find_project_root(&path) {
        let src_dir = root.join("src");
        if src_dir.is_dir() {
            search_paths.push(src_dir);
        }
    }
    if let Ok(env_path) = std::env::var("BORING_PATH") {
        search_paths.extend(std::env::split_paths(&env_path));
    }
    merge_into(&path, &mut visited, &mut items, &search_paths);
    ast::Program { items }
}

fn merge_into(
    path: &Path,
    visited: &mut std::collections::HashSet<PathBuf>,
    items: &mut Vec<ast::Item>,
    search_paths: &[PathBuf],
) {
    let canonical = match path.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: cannot resolve '{}': {}", path.display(), e);
            process::exit(1);
        }
    };
    if !visited.insert(canonical.clone()) {
        return; // already merged (circular or duplicate `use`)
    }

    let source = match std::fs::read_to_string(&canonical) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read '{}': {}", canonical.display(), e);
            process::exit(1);
        }
    };

    let tokens = match lexer::lex_all(&source) {
        Ok(t) => t,
        Err(errors) => {
            report_lex_errors(&canonical, &source, &errors);
            process::exit(1);
        }
    };

    let program = match parser::parse(tokens) {
        Ok(p) => p,
        Err(e) => {
            report_error(&canonical, &source, e.line(), e.col(), e.len(), &e.msg());
            process::exit(1);
        }
    };

    let dir = canonical.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
    for item in program.items {
        if let ast::Item::Use(u) = &item {
            let rel: PathBuf = u.path.iter().collect::<PathBuf>().with_extension("br");
            let candidate = std::iter::once(&dir)
                .chain(search_paths.iter())
                .map(|base| base.join(&rel))
                .find(|c| c.exists());
            if let Some(candidate) = candidate {
                merge_into(&candidate, visited, items, search_paths);
                continue; // inlined -- the `use` item itself is now redundant
            }
            // Not a boring source file (e.g. `use std.collections`) -- keep the
            // item as-is so callers that also run the general transpiler over
            // this merged program (which knows how to emit real Rust `use`
            // statements for external crates) still see it.
        }
        items.push(item);
    }
}

fn emit_cuda(path: &str, version: &str) {
    let program = parse_and_merge_program(path);
    let path = PathBuf::from(path);

    let stem = path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let base_dir    = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let project_dir = base_dir.join(format!("{}_cuda", stem));

    let cuda_out = transpiler::cuda::transpile_cuda(&program, &stem, version);
    if !cuda_out.errors.is_empty() {
        let source = std::fs::read_to_string(&path).unwrap_or_default();
        report_transpile_errors(&path, &source, &cuda_out.errors);
        process::exit(1);
    }

    // Create directory layout.
    let src_dir     = project_dir.join("src");
    let kernels_dir = project_dir.join("kernels");
    for dir in [&src_dir, &kernels_dir] {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("error: cannot create '{}': {}", dir.display(), e);
            process::exit(1);
        }
    }

    // src/main.rs
    let main_rs = src_dir.join("main.rs");
    if let Err(e) = std::fs::write(&main_rs, &cuda_out.host_rs) {
        eprintln!("error: cannot write '{}': {}", main_rs.display(), e);
        process::exit(1);
    }

    // kernels/main.cu
    let cu_file = kernels_dir.join("main.cu");
    if let Err(e) = std::fs::write(&cu_file, &cuda_out.device_cu) {
        eprintln!("error: cannot write '{}': {}", cu_file.display(), e);
        process::exit(1);
    }

    // build.rs
    let build_rs = project_dir.join("build.rs");
    if let Err(e) = std::fs::write(&build_rs, &cuda_out.build_rs) {
        eprintln!("error: cannot write '{}': {}", build_rs.display(), e);
        process::exit(1);
    }

    // Cargo.toml
    let cargo_toml = project_dir.join("Cargo.toml");
    if let Err(e) = std::fs::write(&cargo_toml, &cuda_out.cargo_toml) {
        eprintln!("error: cannot write '{}': {}", cargo_toml.display(), e);
        process::exit(1);
    }

    eprintln!("Generated CUDA project at '{}'", project_dir.display());
    eprintln!("  Requires CUDA toolkit (nvcc) and a CUDA-capable GPU.");
    eprintln!("  cd {} && cargo build", project_dir.display());
    if !cuda_out.kernel_names.is_empty() {
        eprintln!("  Kernels: {}", cuda_out.kernel_names.join(", "));
    }
}

fn emit_metal(path: &str, version: &str) {
    let program = parse_and_merge_program(path);
    let path = PathBuf::from(path);

    let stem = path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let base_dir    = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let project_dir = base_dir.join(format!("{}_metal", stem));

    let metal_out = transpiler::metal::transpile_metal(&program, &stem, version);
    if !metal_out.errors.is_empty() {
        let source = std::fs::read_to_string(&path).unwrap_or_default();
        report_transpile_errors(&path, &source, &metal_out.errors);
        process::exit(1);
    }

    // Create directory layout.
    let src_dir     = project_dir.join("src");
    let kernels_dir = project_dir.join("kernels");
    for dir in [&src_dir, &kernels_dir] {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("error: cannot create '{}': {}", dir.display(), e);
            process::exit(1);
        }
    }

    // src/main.rs
    let main_rs = src_dir.join("main.rs");
    if let Err(e) = std::fs::write(&main_rs, &metal_out.host_rs) {
        eprintln!("error: cannot write '{}': {}", main_rs.display(), e);
        process::exit(1);
    }

    // kernels/main.metal
    let metal_file = kernels_dir.join("main.metal");
    if let Err(e) = std::fs::write(&metal_file, &metal_out.device_msl) {
        eprintln!("error: cannot write '{}': {}", metal_file.display(), e);
        process::exit(1);
    }

    // Cargo.toml (no build.rs â€” MSL compiled at runtime via newLibraryWithSource)
    let cargo_toml = project_dir.join("Cargo.toml");
    if let Err(e) = std::fs::write(&cargo_toml, &metal_out.cargo_toml) {
        eprintln!("error: cannot write '{}': {}", cargo_toml.display(), e);
        process::exit(1);
    }

    eprintln!("Generated Metal project at '{}'", project_dir.display());
    eprintln!("  Requires macOS 11+ with Metal-capable GPU.");
    eprintln!("  cd {} && cargo build", project_dir.display());
    if !metal_out.kernel_names.is_empty() {
        eprintln!("  Kernels: {}", metal_out.kernel_names.join(", "));
    }
}

fn emit_wgpu(path: &str, version: &str) {
    let program = parse_and_merge_program(path);
    let path = PathBuf::from(path);

    let stem = path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let base_dir    = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let project_dir = base_dir.join(format!("{}_wgpu", stem));

    let wgpu_out = transpiler::wgpu::transpile_wgpu(&program, &stem, version);
    if !wgpu_out.errors.is_empty() {
        // Best-effort source for pretty-printing (line/col + a caret under the
        // offending text) -- accurate for errors in the entry file itself; an error
        // originating in a `use`-imported file (parse_and_merge_program flattens
        // several files into one Program) will show against the wrong text, but
        // still reports the right message, line, and column.
        let source = std::fs::read_to_string(&path).unwrap_or_default();
        report_transpile_errors(&path, &source, &wgpu_out.errors);
        process::exit(1);
    }

    // Create directory layout.
    let src_dir     = project_dir.join("src");
    let shaders_dir = project_dir.join("shaders");
    for dir in [&src_dir, &shaders_dir] {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("error: cannot create '{}': {}", dir.display(), e);
            process::exit(1);
        }
    }

    // src/main.rs
    let main_rs = src_dir.join("main.rs");
    if let Err(e) = std::fs::write(&main_rs, &wgpu_out.host_rs) {
        eprintln!("error: cannot write '{}': {}", main_rs.display(), e);
        process::exit(1);
    }

    // shaders/main.wgsl
    let wgsl_file = shaders_dir.join("main.wgsl");
    if let Err(e) = std::fs::write(&wgsl_file, &wgpu_out.device_wgsl) {
        eprintln!("error: cannot write '{}': {}", wgsl_file.display(), e);
        process::exit(1);
    }

    // Cargo.toml
    let cargo_toml = project_dir.join("Cargo.toml");
    if let Err(e) = std::fs::write(&cargo_toml, &wgpu_out.cargo_toml) {
        eprintln!("error: cannot write '{}': {}", cargo_toml.display(), e);
        process::exit(1);
    }

    eprintln!("Generated wgpu project at '{}'", project_dir.display());
    eprintln!("  Requires a DirectX 12 (Windows), Vulkan (Windows/Linux), or Metal (macOS) capable GPU.");
    eprintln!("  cd {} && cargo build", project_dir.display());
    if !wgpu_out.kernel_names.is_empty() {
        eprintln!("  Kernels: {}", wgpu_out.kernel_names.join(", "));
    }
}
fn emit_kernel(path: &str) {
    emit_kernel_with_version(path, "0.1.0");
}

fn emit_kernel_with_version(path: &str, version: &str) {
    let path = PathBuf::from(path);

    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read '{}': {}", path.display(), e);
            process::exit(1);
        }
    };

    let tokens = match lexer::lex_all(&source) {
        Ok(t) => t,
        Err(errors) => {
            report_lex_errors(&path, &source, &errors);
            process::exit(1);
        }
    };

    let program = match parser::parse(tokens) {
        Ok(p) => p,
        Err(e) => {
            report_error(&path, &source, e.line(), e.col(), e.len(), &e.msg());
            process::exit(1);
        }
    };

    // Validate for kernel-mode compatibility.
    let diags = validator::validate_kernel(&program);
    let mut has_errors = false;
    for diag in &diags {
        match diag.level {
            validator::DiagLevel::Warning => eprintln!("warning: {}", diag.message),
            validator::DiagLevel::Error   => {
                eprintln!("error: {}", diag.message);
                has_errors = true;
            }
        }
    }
    if has_errors {
        process::exit(1);
    }

    let kernel_out = transpiler::kernel::transpile_kernel(&program);
    let rust_code  = kernel_out.code;

    // Determine output project directory next to the source file.
    let stem = path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let base_dir    = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let project_dir = base_dir.join(format!("{}_kernel", stem));

    // Create directory layout.
    let src_dir = project_dir.join("src");
    if let Err(e) = std::fs::create_dir_all(&src_dir) {
        eprintln!("error: cannot create '{}': {}", src_dir.display(), e);
        process::exit(1);
    }

    // Write src/lib.rs (kernel modules are library crates).
    let lib_rs = src_dir.join("lib.rs");
    if let Err(e) = std::fs::write(&lib_rs, rust_code) {
        eprintln!("error: cannot write '{}': {}", lib_rs.display(), e);
        process::exit(1);
    }

    // Write Cargo.toml â€” no tokio; the kernel crate is provided by the build system.
    let cargo_toml = format!(
        r#"[package]
name = "{stem}"
version = "{version}"
edition = "2024"

[lib]
name = "{stem}"
path = "src/lib.rs"

[dependencies]
# kernel crate is provided by the build system (Linux kernel Rust infrastructure)
"#
    );
    let cargo_path = project_dir.join("Cargo.toml");
    if let Err(e) = std::fs::write(&cargo_path, cargo_toml) {
        eprintln!("error: cannot write '{}': {}", cargo_path.display(), e);
        process::exit(1);
    }

    eprintln!("Generated kernel Cargo project at '{}'", project_dir.display());
    eprintln!("  Build with the Linux kernel build system (make -C /path/to/linux M=$PWD)");
}


