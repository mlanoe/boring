// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of Boring.
// Boring is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// See the LICENSE file at the project root for the full text.

pub mod lexer;
pub mod ast;
pub mod parser;
pub mod interpreter;
pub mod checker;
pub mod transpiler;
pub mod validator;

use std::path::{Path, PathBuf};
use std::process;

// ─── Diagnostics ─────────────────────────────────────────────────────────────

/// Print a Rust-style diagnostic with source context.
///
/// ```text
/// error: <message>
///  --> path/to/file.br:5
///   |
/// 5 | let z = x + y
///   |
/// ```
fn report_error(path: &Path, source: &str, line: usize, message: &str) {
    eprintln!("error: {}", message);
    if line == 0 {
        eprintln!(" --> {}", path.display());
        return;
    }
    eprintln!(" --> {}:{}", path.display(), line);
    let width = line.to_string().len();
    let pad   = " ".repeat(width);
    eprintln!("{} |", pad);
    let src_line = source.lines().nth(line.saturating_sub(1)).unwrap_or("");
    eprintln!("{} | {}", line, src_line);
    eprintln!("{} |", pad);
}

// ─── boring.toml ─────────────────────────────────────────────────────────────

/// Minimal `boring.toml` representation.
struct BoringToml {
    name:    String,
    version: String,
    main:    String,
}

impl BoringToml {
    /// Parse a `boring.toml` file.  No external dependency — the format is tiny.
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

// ─── Entry point ─────────────────────────────────────────────────────────────

fn main() {
    // Windows default stack is 1 MB — too small for the interpreter's recursive
    // descent (expression evaluation, type resolution, pattern matching).
    // Spawn a new thread with an 8 MB stack so the same code runs everywhere.
    const STACK_SIZE: usize = 8 * 1024 * 1024; // 8 MB
    let builder = std::thread::Builder::new().stack_size(STACK_SIZE);
    let handler = builder.spawn(run).expect("failed to spawn main thread");
    match handler.join() {
        Ok(()) => {}
        Err(e) => {
            // The worker thread panicked — print the payload for diagnosis.
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

fn run() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("--help") | Some("-h") => {
            print_help();
            process::exit(0);
        }

        // ── Project commands ────────────────────────────────────────────────
        Some("new") => {
            let name = args.get(2).unwrap_or_else(|| {
                eprintln!("usage: boring new <name>");
                process::exit(1);
            });
            new_project(name);
        }
        Some("run") => {
            match args.get(2).map(|s| s.as_str()) {
                Some(path) => run_file(path),         // boring run file.br
                None       => run_project(),          // boring run  (uses boring.toml)
            }
        }
        Some("build") => {
            let build_args = &args[2..];
            parse_build_command(build_args);
        }

        Some(path) => run_file(path),
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
    eprintln!("    boring <file.br>           Run a single file (shorthand)");
}

// ─── Project commands ─────────────────────────────────────────────────────────

/// `boring new <name>` — scaffold a project directory.
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

/// `boring run` — run the project described by `boring.toml` in the current directory.
fn run_project() {
    let (toml, _) = load_project_toml();
    run_file(&toml.main);
}

/// `boring build` — emit a Cargo project from the `boring.toml` main file.

fn build_project_with_config(config: transpiler::TranspileConfig) {
    let (toml, _) = load_project_toml();
    emit_rust_with_version_and_config(&toml.main, &toml.version, config);
}

/// `boring build --target kernel` — emit a kernel Cargo project from `boring.toml`.
fn build_project_kernel() {
    let (toml, _) = load_project_toml();
    emit_kernel_with_version(&toml.main, &toml.version);
}

/// Parse `boring build [flags] [file.br]` arguments after the `build` subcommand.
fn parse_build_command(build_args: &[String]) {
    use transpiler::{TranspileConfig, TranspileMode, ThreadingMode};

    let mut target_kernel = false;
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
                    Some(t) => {
                        eprintln!("error: unknown target '{}'", t);
                        eprintln!("hint:  supported targets: kernel");
                        process::exit(1);
                    }
                    None => {
                        eprintln!("error: --target requires a value");
                        eprintln!("hint:  supported targets: kernel");
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
                        eprintln!("error: unknown mode '{}' — expected strict or managed", m);
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
                        eprintln!("error: unknown threading model '{}' — expected single or multi", t);
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
                        eprintln!("error: unknown sanitizer '{}' — expected address, thread, or memory", s);
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

    let config = TranspileConfig { mode, threading, stack_auto_bytes, instrument, sanitize, source_dir: PathBuf::new() };

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
            let dir = base.join(rust_dir_name(&stem, &config.threading));
            emit_rust_with_config(path, config);
            dir
        }
        (false, None, _)                => {
            let (toml, _) = load_project_toml();
            let stem = std::path::Path::new(&toml.main)
                .file_stem().map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "output".to_string());
            let base = std::path::Path::new(&toml.main).parent().unwrap_or(std::path::Path::new("."));
            let dir = base.join(rust_dir_name(&stem, &config.threading));
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

// ─── Core: interpret ──────────────────────────────────────────────────────────

fn run_file(path: &str) {
    let path = PathBuf::from(path);

    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read '{}': {}", path.display(), e);
            process::exit(1);
        }
    };

    let tokens = match lexer::lex(&source) {
        Ok(t) => t,
        Err(e) => {
            report_error(&path, &source, e.line(), &e.msg());
            process::exit(1);
        }
    };

    let program = match parser::parse(tokens) {
        Ok(p) => p,
        Err(e) => {
            report_error(&path, &source, e.line(), &e.msg());
            process::exit(1);
        }
    };

    let mut interp = interpreter::Interpreter::new();

    // Add the file's directory to the search path for `use` resolution
    if let Some(dir) = path.parent() {
        interp.add_search_path(dir.to_path_buf());
    }

    // Add BORING_PATH entries (uses OS path-list separator: `:` on Unix, `;` on Windows)
    if let Ok(env_path) = std::env::var("BORING_PATH") {
        for p in std::env::split_paths(&env_path) {
            interp.add_search_path(p);
        }
    }

    if let Err(e) = interp.exec_program(&program) {
        report_error(&path, &source, e.line, &e.message);
        process::exit(1);
    }
}

// ─── Core: transpile ──────────────────────────────────────────────────────────

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
    let tokens = match lexer::lex(&source) {
        Ok(t) => t,
        Err(e) => { report_error(&path, &source, e.line(), &e.msg()); process::exit(1); }
    };
    let program = match parser::parse(tokens) {
        Ok(p) => p,
        Err(e) => { report_error(&path, &source, e.line(), &e.msg()); process::exit(1); }
    };
    let out = transpiler::transpile_with_config(&program, config);
    print!("{}", out.code);
}

fn rust_dir_name(stem: &str, threading: &transpiler::ThreadingMode) -> String {
    if matches!(threading, transpiler::ThreadingMode::Single) {
        format!("{}_rust_single", stem)
    } else {
        format!("{}_rust", stem)
    }
}

fn emit_rust_with_version_and_config(path: &str, version: &str, config: transpiler::TranspileConfig) {
    let path = PathBuf::from(path);

    // Determine output project directory next to the source file
    let stem = path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let base_dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let project_dir = base_dir.join(rust_dir_name(&stem, &config.threading));

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

    let tokens = match lexer::lex(&source) {
        Ok(t) => t,
        Err(e) => {
            report_error(&path, &source, e.line(), &e.msg());
            process::exit(1);
        }
    };

    let program = match parser::parse(tokens) {
        Ok(p) => p,
        Err(e) => {
            report_error(&path, &source, e.line(), &e.msg());
            process::exit(1);
        }
    };

    let source_dir = path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
    let config_with_dir = transpiler::TranspileConfig { source_dir, ..config.clone() };
    let transpile_out = transpiler::transpile_with_config(&program, config_with_dir);
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
            eprintln!("note: sanitizer enabled — run with: cargo +nightly run");
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

// ─── Core: kernel transpile ───────────────────────────────────────────────────

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

    let tokens = match lexer::lex(&source) {
        Ok(t) => t,
        Err(e) => {
            report_error(&path, &source, e.line(), &e.msg());
            process::exit(1);
        }
    };

    let program = match parser::parse(tokens) {
        Ok(p) => p,
        Err(e) => {
            report_error(&path, &source, e.line(), &e.msg());
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

    // Write Cargo.toml — no tokio; the kernel crate is provided by the build system.
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
