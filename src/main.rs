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
            match args.get(2).map(|s| s.as_str()) {
                Some(path) => emit_rust(path),        // boring build file.br
                None       => build_project(),        // boring build  (uses boring.toml)
            }
        }

        // ── Legacy / direct-file commands ──────────────────────────────────
        Some("--emit-rust") => {
            let path = args.get(2).unwrap_or_else(|| {
                eprintln!("usage: boring --emit-rust <file.br>");
                process::exit(1);
            });
            emit_rust(path);
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
    eprintln!("    boring <file.br>           Run a single file (shorthand)");
    eprintln!("    boring --emit-rust <file>  Alias for `boring build <file>`");
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
fn build_project() {
    let (toml, _) = load_project_toml();
    emit_rust_with_version(&toml.main, &toml.version);
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

fn emit_rust(path: &str) {
    emit_rust_with_version(path, "0.1.0");
}

fn emit_rust_with_version(path: &str, version: &str) {
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

    let transpile_out = transpiler::transpile_full(&program);
    let rust_code = transpile_out.code;
    let has_streams = transpile_out.has_streams;
    let uses_log = transpile_out.uses_log;
    let uses_thiserror = transpile_out.uses_thiserror;
    let uses_reqwest = transpile_out.uses_reqwest;
    let uses_tokio_util = transpile_out.uses_tokio_util;
    let uses_serde = transpile_out.uses_serde;

    // Determine output project directory next to the source file
    let stem = path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let base_dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let project_dir = base_dir.join(format!("{}_rust", stem));

    // Create directory layout
    let src_dir = project_dir.join("src");
    if let Err(e) = std::fs::create_dir_all(&src_dir) {
        eprintln!("error: cannot create '{}': {}", src_dir.display(), e);
        process::exit(1);
    }

    // Write src/main.rs  (strip the old standalone rustc comment if present)
    let clean_code = rust_code
        .strip_prefix("// Generated by boring --emit-rust. Compile with: rustc --edition 2021 <file.rs>\n")
        .unwrap_or(&rust_code);
    let main_rs = src_dir.join("main.rs");
    if let Err(e) = std::fs::write(&main_rs, clean_code) {
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
    let cargo_toml = format!(
        r#"[package]
name = "{stem}"
version = "{version}"
edition = "2024"

[[bin]]
name = "{stem}"
path = "src/main.rs"

[dependencies]
tokio = {{ version = "1", features = ["full"] }}{stream_deps}{log_dep}{thiserror_dep}{reqwest_dep}{tokio_util_dep}{serde_dep}"#
    );
    let cargo_path = project_dir.join("Cargo.toml");
    if let Err(e) = std::fs::write(&cargo_path, cargo_toml) {
        eprintln!("error: cannot write '{}': {}", cargo_path.display(), e);
        process::exit(1);
    }

    eprintln!("Generated Cargo project at '{}'", project_dir.display());
    eprintln!("  cd {} && cargo run", project_dir.display());
}
