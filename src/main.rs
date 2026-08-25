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
pub mod desugar_labeled_array;
pub mod interpreter;
pub mod checker;
mod git_deps;
pub mod stdlib_embed;
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
    /// Raw `[dependencies]` lines (e.g. `bevy = { version = "0.19", default-features = false }`),
    /// copied verbatim into the generated Cargo.toml's own `[dependencies]` section — see
    /// `Self::parse`'s doc comment for why this stays text, not a real TOML value.
    dependencies: Vec<String>,
    /// `[external_types]` `tuple_structs = [...]` entries — project-declared supplement to
    /// the transpiler's built-in `KNOWN_EXTERNAL_TUPLE_STRUCTS` (see that const's doc
    /// comment in `transpiler/mod.rs`). Threaded into `TranspileConfig::external_tuple_structs`
    /// by `build_project_with_config`.
    external_tuple_structs: Vec<String>,
    /// `[external_types]` `const_fns = [...]` entries, each written `"Type::method"` and
    /// split here into `(type, method)` pairs — project-declared supplement to
    /// `KNOWN_EXTERNAL_CONST_FNS`. Threaded into `TranspileConfig::external_const_fns`.
    external_const_fns: Vec<(String, String)>,
    /// `[external_types]` `include = [...]` entries — paths (relative to this `boring.toml`'s
    /// own directory) to other files with their own `[external_types]` section, folded into
    /// `external_tuple_structs`/`external_const_fns` above by `resolve_external_types_includes`.
    /// Lets several sibling projects (e.g. multiple Bevy games) share one canonical
    /// declarations file instead of each repeating the same entries. Consumed and drained by
    /// `resolve_external_types_includes` — empty again once `load_project_toml` returns.
    external_types_includes: Vec<String>,
    /// `[derives]` `traits = [...]` entries — project-declared supplement to the
    /// transpiler's built-in `KNOWN_DERIVABLE_TRAITS` (see that const's doc comment in
    /// `transpiler/mod.rs`). A name in this list, when it appears in a struct/enum's header
    /// `as Trait1, Trait2:` list, is routed into `#[derive(...)]` instead of requiring a
    /// manual `impl Trait for X { ... }` — e.g. Bevy's `Component`/`Resource`. Threaded into
    /// `TranspileConfig::known_derives` by `build_project_with_config`.
    derive_traits: Vec<String>,
    /// `[derives]` `include = [...]` entries — same shape and semantics as
    /// `external_types_includes`, but for `derive_traits`. Consumed and drained by
    /// `resolve_derive_includes` — empty again once `load_project_toml` returns.
    derive_includes: Vec<String>,
    /// `[deps]` entries — named dependencies on other Boring *projects* (as opposed to
    /// `[dependencies]`'s Rust crates), each `(name, raw_value)`. `raw_value` is kept as
    /// opaque text here (same "the format is tiny" philosophy as `[dependencies]`) and only
    /// interpreted by `resolve_deps`, which turns it into an actual filesystem path (or
    /// rejects it) — see that method's doc comment for the value grammar
    /// (`"../path"` or `{ path = "..." }` or `{ git = "..." }`).
    deps: Vec<(String, String)>,
    /// `[external_fns]` entries — project-declared supplement to the transpiler's built-in
    /// `KNOWN_EXTERNAL_FN_BORROWS` (see that const's doc comment in `transpiler/mod.rs`).
    /// Each line is `"Qualifier::method" = ["&mut", "&", ...]` — `Qualifier` is an external
    /// type name for a method call, or a free function's fully-qualified module path (see
    /// `KNOWN_EXTERNAL_FN_BORROWS`'s doc comment for the exact split); the array is one
    /// borrow-form entry (`"&"`, `"&mut"`, or `""`) per call argument after the receiver.
    /// Parsed here into `(qualifier, method, borrows)` triples by `Self::parse_external_fn_key`
    /// splitting the key on its *last* `::` (a free function's qualifier can itself contain
    /// `::`, e.g. `"std::mem::swap"` splits to qualifier `"std::mem"`, method `"swap"`).
    /// Threaded into `TranspileConfig::external_fns`.
    external_fns: Vec<(String, String, Vec<String>)>,
    /// `[external_fns]` `include = [...]` entries — same shape and semantics as
    /// `external_types_includes`/`derive_includes`, but for `external_fns`. Consumed and
    /// drained by `resolve_external_fns_includes` — empty again once `load_project_toml`
    /// returns.
    external_fns_includes: Vec<String>,
}

/// One `[deps]` entry's interpreted value — see `BoringToml::resolve_deps`.
#[derive(Debug, PartialEq)]
enum DepSpec {
    Path(String),
    Git { url: String, gitref: GitRef },
}

/// Which ref of a `git` dependency to check out — see `git_deps::resolve_git_dep`.
/// `Default` means "whatever the remote's default branch is" (a plain `git clone`
/// with no `--branch`). Exactly one of `branch`/`tag`/`rev` may be given in a
/// `{ git = "...", ... }` table; giving more than one is a `parse_dep_value` error
/// since it's ambiguous which one should win.
#[derive(Debug, PartialEq, Clone)]
enum GitRef {
    Default,
    Branch(String),
    Tag(String),
    Rev(String),
}

impl BoringToml {
    /// Parse a `boring.toml` file.  No external dependency â€” the format is tiny: flat
    /// `key = value` lines, plus two recognized section headers. `[dependencies]`'s body
    /// lines are kept as opaque text (not parsed) and spliced verbatim into the generated
    /// Cargo.toml's `[dependencies]` section by `emit_rust_to_dir`. This is the only way
    /// today to add an external Cargo dependency (e.g. `bevy`) without hand-editing the
    /// generated project after `boring build` — Boring itself has no opinion on the
    /// value's syntax (inline tables, feature arrays, etc. all pass through as-is).
    /// `[external_types]`'s recognized keys (`tuple_structs`/`const_fns`, each a plain
    /// inline string array, plus `include`, an array of paths to other files with their own
    /// `[external_types]` section — see `resolve_external_types_includes`) *are* interpreted,
    /// unlike `[dependencies]` — they're project-declared supplements to the transpiler's own
    /// hand-verified `KNOWN_EXTERNAL_TUPLE_STRUCTS`/`KNOWN_EXTERNAL_CONST_FNS` (see those
    /// consts' doc comments in `transpiler/mod.rs`), which the transpiler needs as actual data
    /// (a type name, or a `(type, method)` pair), not opaque text to copy elsewhere. Always
    /// additive: a project can only add entries, never remove or override a built-in one.
    /// `[derives]` (`traits`/`include`) follows the exact same shape and rationale, for
    /// `KNOWN_DERIVABLE_TRAITS` instead — see `resolve_derive_includes`.
    fn parse(src: &str) -> Self {
        let mut name    = String::new();
        let mut version = "0.1.0".to_string();
        let mut main    = "main.br".to_string();
        let mut dependencies = Vec::new();
        let mut external_tuple_structs = Vec::new();
        let mut external_const_fns = Vec::new();
        let mut external_types_includes = Vec::new();
        let mut derive_traits = Vec::new();
        let mut derive_includes = Vec::new();
        let mut deps = Vec::new();
        let mut external_fns = Vec::new();
        let mut external_fns_includes = Vec::new();
        let mut in_dependencies = false;
        let mut in_external_types = false;
        let mut in_derives = false;
        let mut in_deps = false;
        let mut in_external_fns = false;

        for line in src.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with('[') {
                in_dependencies = line == "[dependencies]";
                in_external_types = line == "[external_types]";
                in_derives = line == "[derives]";
                in_deps = line == "[deps]";
                in_external_fns = line == "[external_fns]";
                continue;
            }
            if in_dependencies {
                dependencies.push(line.to_string());
            } else if in_deps {
                // Same one-line-per-entry style as `[dependencies]`, but keyed: split on the
                // first `=` into (name, raw_value). `resolve_deps` interprets `raw_value`.
                if let Some((name, value)) = line.split_once('=') {
                    deps.push((name.trim().to_string(), value.trim().to_string()));
                }
            } else if in_external_fns {
                if let Some(rest) = line.strip_prefix("include") {
                    external_fns_includes = Self::extract_array(rest);
                } else if let Some((key, value)) = line.split_once('=') {
                    // `"Qualifier::method" = ["&mut", "&mut"]` — key is a quoted string; the
                    // value is a positional array, so — unlike every other `extract_array`
                    // call site in this file (`tuple_structs`/`const_fns`/`traits`/`include`,
                    // where a blank entry never carries meaning) — an empty `""` element here
                    // is a real, meaningful "by value, no borrow" placeholder that MUST keep
                    // its position (e.g. `["&", "", "&mut"]` for `resvg::render(tree: &Tree,
                    // transform: Transform, pixmap: &mut PixmapMut)` — the by-value `transform`
                    // in the middle). `extract_array`'s own `.filter(|s| !s.is_empty())` would
                    // silently drop that element and shift every later position left — found
                    // via exactly this `resvg::render` real-crate repro, where it turned
                    // `["&", "", "&mut"]` into `["&", "&mut"]` and mis-borrowed both later
                    // arguments. `extract_borrow_array` below is `extract_array` without that
                    // filter — used only here, never for the other whitelist tables.
                    let key = key.trim().trim_matches('"');
                    if let Some((qualifier, method)) = Self::parse_external_fn_key(key) {
                        external_fns.push((qualifier, method, Self::extract_borrow_array(value)));
                    }
                }
            } else if in_external_types {
                if let Some(rest) = line.strip_prefix("tuple_structs") {
                    external_tuple_structs = Self::extract_array(rest);
                } else if let Some(rest) = line.strip_prefix("const_fns") {
                    external_const_fns = Self::extract_array(rest)
                        .into_iter()
                        .filter_map(|entry| entry.split_once("::").map(|(t, m)| (t.to_string(), m.to_string())))
                        .collect();
                } else if let Some(rest) = line.strip_prefix("include") {
                    external_types_includes = Self::extract_array(rest);
                }
            } else if in_derives {
                if let Some(rest) = line.strip_prefix("traits") {
                    derive_traits = Self::extract_array(rest);
                } else if let Some(rest) = line.strip_prefix("include") {
                    derive_includes = Self::extract_array(rest);
                }
            } else if let Some(rest) = line.strip_prefix("name") {
                if let Some(v) = Self::extract_value(rest) { name = v; }
            } else if let Some(rest) = line.strip_prefix("version") {
                if let Some(v) = Self::extract_value(rest) { version = v; }
            } else if let Some(rest) = line.strip_prefix("main") {
                if let Some(v) = Self::extract_value(rest) { main = v; }
            }
        }
        BoringToml {
            name, version, main, dependencies,
            external_tuple_structs, external_const_fns, external_types_includes,
            derive_traits, derive_includes, deps,
            external_fns, external_fns_includes,
        }
    }

    /// Splits an `[external_fns]` key on its *last* `::` into `(qualifier, method)` — e.g.
    /// `"std::mem::swap"` → `("std::mem", "swap")`, `"ZipFile::readToEnd"` →
    /// `("ZipFile", "readToEnd")`. Unlike `[external_types]`'s `const_fns` (which splits
    /// `"Type::method"` on the *first* `::` since a type name never itself contains `::`), a
    /// free function's qualifier is a full module path that can contain further `::` of its
    /// own, so only the last segment can safely be assumed to be the method name. Returns
    /// `None` for a key with no `::` at all (silently skipped, same "malformed entry is
    /// dropped rather than a hard parse error" convention `const_fns`'s `filter_map` already
    /// uses).
    fn parse_external_fn_key(key: &str) -> Option<(String, String)> {
        key.rsplit_once("::").map(|(q, m)| (q.to_string(), m.to_string()))
    }

    /// Resolves this `boring.toml`'s `[external_types]` `include` paths (each relative to
    /// `boring_toml_dir`, i.e. this `boring.toml`'s own directory), folding each included
    /// file's own `tuple_structs`/`const_fns` into this project's own lists — this is what
    /// lets several sibling projects (e.g. multiple Bevy games) share one canonical
    /// declarations file instead of each repeating the same entries (see `boring-bevylib`'s
    /// `external_types.toml` for a real example). Single-level only: an included file's own
    /// `include` key, if it has one, is silently ignored rather than followed recursively —
    /// deliberately simple, no cycle detection needed as a result. Drains
    /// `external_types_includes` as it goes (idempotent: calling this twice is a no-op the
    /// second time). Returns an error message (rather than exiting directly) on the first
    /// unreadable include path, so this stays independently unit-testable.
    fn resolve_external_types_includes(&mut self, boring_toml_dir: &Path) -> Result<(), String> {
        for include_path in std::mem::take(&mut self.external_types_includes) {
            let resolved = boring_toml_dir.join(&include_path);
            let included_src = std::fs::read_to_string(&resolved).map_err(|e| {
                format!("cannot read external_types include '{}': {}", resolved.display(), e)
            })?;
            let included = Self::parse(&included_src);
            self.external_tuple_structs.extend(included.external_tuple_structs);
            self.external_const_fns.extend(included.external_const_fns);
        }
        Ok(())
    }

    /// Resolves this `boring.toml`'s `[derives]` `include` paths the same way
    /// `resolve_external_types_includes` resolves `[external_types]`'s — same relative-to-dir
    /// resolution, same single-level (no recursion into a nested `include`), same
    /// additive-only fold into `derive_traits`, same idempotent drain-based design, and same
    /// `Result<(), String>` return so this stays independently unit-testable rather than
    /// exiting directly.
    fn resolve_derive_includes(&mut self, boring_toml_dir: &Path) -> Result<(), String> {
        for include_path in std::mem::take(&mut self.derive_includes) {
            let resolved = boring_toml_dir.join(&include_path);
            let included_src = std::fs::read_to_string(&resolved).map_err(|e| {
                format!("cannot read derives include '{}': {}", resolved.display(), e)
            })?;
            let included = Self::parse(&included_src);
            self.derive_traits.extend(included.derive_traits);
        }
        Ok(())
    }

    /// Resolves this `boring.toml`'s `[external_fns]` `include` paths the same way
    /// `resolve_external_types_includes`/`resolve_derive_includes` do — same relative-to-dir
    /// resolution, same single-level (no recursion into a nested `include`), same
    /// additive-only fold into `external_fns`, same idempotent drain-based design, and same
    /// `Result<(), String>` return so this stays independently unit-testable rather than
    /// exiting directly.
    fn resolve_external_fns_includes(&mut self, boring_toml_dir: &Path) -> Result<(), String> {
        for include_path in std::mem::take(&mut self.external_fns_includes) {
            let resolved = boring_toml_dir.join(&include_path);
            let included_src = std::fs::read_to_string(&resolved).map_err(|e| {
                format!("cannot read external_fns include '{}': {}", resolved.display(), e)
            })?;
            let included = Self::parse(&included_src);
            self.external_fns.extend(included.external_fns);
        }
        Ok(())
    }

    /// Resolves this `boring.toml`'s `[deps]` entries — named dependencies on other Boring
    /// *projects* (`use <name>.xxx` resolving to `<name>`'s own `src/` directory), as opposed
    /// to `[dependencies]`'s Rust crates. See `docs/cross-project-code-sharing-gap.md` for the
    /// motivation and `docs/book.md` §15 for the user-facing syntax. Each entry's raw value is
    /// either a bare path string (`numlib = "../boring-numlib"`) or a small inline table
    /// (`numlib = { path = "../boring-numlib" }` / `{ git = "..." }`) — parsed by
    /// `parse_dep_value` below, which is deliberately not a real TOML-inline-table parser
    /// (single level, string values only — matches this whole file's "the format is tiny"
    /// philosophy). Resolution rules, checked for every entry regardless of value shape:
    /// - `name` may not be `std`, `crate`, or `boring` — those are the compiler's own reserved
    ///   `use` prefixes (Rust stdlib, current crate, and the first-party embedded stdlib
    ///   respectively) and can never be shadowed by a project dependency.
    /// - `git = "..."` clones the repo into a persistent local cache the first time it's
    ///   needed and reuses/refreshes it on later resolutions — see `git_deps::resolve_git_dep`
    ///   for the caching/refresh strategy. An optional `branch`/`tag`/`rev` key picks which ref
    ///   to check out (default: the remote's default branch).
    /// - A `path` value resolves to `<boring_toml_dir>/<path>/src` (the dependency is just a
    ///   directory with a `src/` convention, not required to have its own `boring.toml`) —
    ///   relative to *this* `boring.toml`'s own directory, same as
    ///   `resolve_external_types_includes`. No transitive resolution: if that directory has
    ///   its own `boring.toml` with its own `[deps]`, those are not followed — same
    ///   single-level rule `resolve_external_types_includes`/`resolve_derive_includes` already
    ///   follow for `include`.
    ///   Returns an error message (rather than exiting directly) on the first invalid entry, so
    ///   this stays independently unit-testable, same convention as the `include` resolvers.
    fn resolve_deps(&self, boring_toml_dir: &Path) -> Result<std::collections::HashMap<String, PathBuf>, String> {
        let mut resolved = std::collections::HashMap::new();
        for (name, raw_value) in &self.deps {
            if matches!(name.as_str(), "std" | "crate" | "boring") {
                return Err(format!(
                    "'{}' is a reserved name and cannot be used as a [deps] dependency name",
                    name
                ));
            }
            match Self::parse_dep_value(raw_value)? {
                DepSpec::Path(p) => {
                    resolved.insert(name.clone(), boring_toml_dir.join(p).join("src"));
                }
                DepSpec::Git { url, gitref } => {
                    let src = crate::git_deps::resolve_git_dep(&url, &gitref)
                        .map_err(|e| format!("dep '{}': {}", name, e))?;
                    resolved.insert(name.clone(), src);
                }
            }
        }
        Ok(resolved)
    }

    /// Parses one `[deps]` entry's raw value — either a quoted path string, or a small
    /// `{ key = "value", ... }` inline table looking for `path` or `git` (+ optional
    /// `branch`/`tag`/`rev`) keys. Not a general TOML inline-table parser: single level,
    /// string values only, no nested tables/arrays — matches `extract_value`/`extract_array`'s
    /// existing "the format is tiny" style. Unlike the old first-match-wins version, this reads
    /// every key in the table before deciding, since `git` needs to see a possible
    /// `branch`/`tag`/`rev` key alongside it regardless of which order they're written in.
    fn parse_dep_value(raw: &str) -> Result<DepSpec, String> {
        let raw = raw.trim();
        if let Some(stripped) = raw.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
            let mut path = None;
            let mut git = None;
            let mut branch = None;
            let mut tag = None;
            let mut rev = None;
            for pair in stripped.split(',') {
                let pair = pair.trim();
                if pair.is_empty() { continue; }
                let Some((key, value)) = pair.split_once('=') else { continue };
                let value = value.trim().trim_matches('"').to_string();
                match key.trim() {
                    "path"   => path = Some(value),
                    "git"    => git = Some(value),
                    "branch" => branch = Some(value),
                    "tag"    => tag = Some(value),
                    "rev"    => rev = Some(value),
                    _ => {}
                }
            }
            if path.is_some() && git.is_some() {
                return Err(format!("[deps] entry '{}' has both 'path' and 'git' — pick one", raw));
            }
            if let Some(p) = path {
                return Ok(DepSpec::Path(p));
            }
            if let Some(url) = git {
                let gitref = match (branch, tag, rev) {
                    (None, None, None) => GitRef::Default,
                    (Some(b), None, None) => GitRef::Branch(b),
                    (None, Some(t), None) => GitRef::Tag(t),
                    (None, None, Some(r)) => GitRef::Rev(r),
                    _ => return Err(format!(
                        "[deps] entry '{}' gives more than one of branch/tag/rev — pick one", raw
                    )),
                };
                return Ok(DepSpec::Git { url, gitref });
            }
            return Err(format!("[deps] entry '{}' has no recognized 'path' or 'git' key", raw));
        }
        if raw.starts_with('"') {
            return Ok(DepSpec::Path(raw.trim_matches('"').to_string()));
        }
        Err(format!("[deps] entry '{}' is neither a quoted path nor a {{ path/git = ... }} table", raw))
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

    /// Extract a plain inline string array from `= ["a", "b"]` — used for
    /// `[external_types]`'s `tuple_structs`/`const_fns` keys. Not general TOML array
    /// parsing (no nesting, no non-string elements) — matches this parser's "the format is
    /// tiny" philosophy (see `Self::parse`'s doc comment).
    fn extract_array(rest: &str) -> Vec<String> {
        let rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == '=').trim();
        let inner = rest.trim_start_matches('[').trim_end_matches(']');
        inner
            .split(',')
            .map(|s| s.trim().trim_matches('"').to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// `extract_array` without its `.filter(|s| !s.is_empty())` — for `[external_fns]`'s
    /// per-argument borrow-form arrays only, where a bare `""` element is a meaningful "by
    /// value" placeholder whose *position* matters (see the call site's own doc comment for
    /// why dropping it silently misaligns every later argument). Every other bracketed-array
    /// value in `boring.toml` (`tuple_structs`, `const_fns`, `traits`, any `include`) is an
    /// unordered set of names, where an accidental blank/trailing-comma entry is noise to
    /// discard, not signal — `extract_array` keeps its filter for those.
    fn extract_borrow_array(rest: &str) -> Vec<String> {
        let rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == '=').trim();
        let inner = rest.trim_start_matches('[').trim_end_matches(']').trim();
        // A genuinely empty array (`= []`, a zero-argument call) must stay `vec![]`, not the
        // single-blank-element `vec![""]` that `"".split(',')` would otherwise produce — that
        // single blank would then be misread as "argument 0 is by value" for a call that
        // actually takes no arguments at all.
        if inner.is_empty() {
            return Vec::new();
        }
        inner
            .split(',')
            .map(|s| s.trim().trim_matches('"').to_string())
            .collect()
    }
}

#[cfg(test)]
mod boring_toml_tests {
    use super::{BoringToml, DepSpec, GitRef};
    use std::path::PathBuf;

    #[test]
    fn dependencies_section_captured_verbatim() {
        let src = "[project]\nname = \"demo\"\nversion = \"0.1.0\"\n\n[dependencies]\nbevy = { version = \"0.19\", default-features = false }\nserde = { version = \"1\", features = [\"derive\"] }\n";
        let toml = BoringToml::parse(src);
        assert_eq!(toml.name, "demo");
        assert_eq!(toml.dependencies, vec![
            "bevy = { version = \"0.19\", default-features = false }".to_string(),
            "serde = { version = \"1\", features = [\"derive\"] }".to_string(),
        ]);
    }

    #[test]
    fn no_dependencies_section_is_empty() {
        let src = "[project]\nname = \"demo\"\nversion = \"0.1.0\"\n";
        let toml = BoringToml::parse(src);
        assert!(toml.dependencies.is_empty());
    }

    #[test]
    fn dependencies_before_project_section_still_parses_name() {
        // Section order shouldn't matter — `name`/`version`/`main` are read
        // whenever we're not inside `[dependencies]`, regardless of position.
        let src = "[dependencies]\nbevy = \"0.19\"\n\n[project]\nname = \"demo\"\n";
        let toml = BoringToml::parse(src);
        assert_eq!(toml.name, "demo");
        assert_eq!(toml.dependencies, vec!["bevy = \"0.19\"".to_string()]);
    }

    #[test]
    fn external_types_section_parses_tuple_structs_and_const_fns() {
        let src = "[project]\nname = \"demo\"\n\n[external_types]\ntuple_structs = [\"Mesh2d\", \"OuterColor\"]\nconst_fns = [\"Vec3::new\", \"Duration::from_secs\"]\n";
        let toml = BoringToml::parse(src);
        assert_eq!(toml.external_tuple_structs, vec!["Mesh2d".to_string(), "OuterColor".to_string()]);
        assert_eq!(toml.external_const_fns, vec![
            ("Vec3".to_string(), "new".to_string()),
            ("Duration".to_string(), "from_secs".to_string()),
        ]);
    }

    #[test]
    fn no_external_types_section_is_empty() {
        let src = "[project]\nname = \"demo\"\n";
        let toml = BoringToml::parse(src);
        assert!(toml.external_tuple_structs.is_empty());
        assert!(toml.external_const_fns.is_empty());
    }

    #[test]
    fn external_types_const_fns_without_double_colon_are_skipped() {
        // A malformed entry (no `Type::method` shape) is silently dropped rather than
        // panicking or corrupting the pair list — same "always-valid fallback over an
        // optimistic guess" philosophy as `is_known_external_const_fn` itself.
        let src = "[external_types]\nconst_fns = [\"NotAPair\", \"Vec3::new\"]\n";
        let toml = BoringToml::parse(src);
        assert_eq!(toml.external_const_fns, vec![("Vec3".to_string(), "new".to_string())]);
    }

    #[test]
    fn external_types_include_parses_to_raw_paths() {
        let src = "[external_types]\ninclude = [\"../shared/external_types.toml\"]\ntuple_structs = [\"LocalOnly\"]\n";
        let toml = BoringToml::parse(src);
        assert_eq!(toml.external_types_includes, vec!["../shared/external_types.toml".to_string()]);
        assert_eq!(toml.external_tuple_structs, vec!["LocalOnly".to_string()]);
    }

    #[test]
    fn resolve_external_types_includes_folds_in_shared_declarations() {
        let dir = std::env::temp_dir().join(format!(
            "boring_toml_include_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("shared_external_types.toml"),
            "[external_types]\ntuple_structs = [\"Shared1\", \"Shared2\"]\nconst_fns = [\"Vec3::new\"]\n",
        ).unwrap();

        let mut toml = BoringToml::parse(
            "[external_types]\ninclude = [\"shared_external_types.toml\"]\ntuple_structs = [\"LocalOnly\"]\n",
        );
        toml.resolve_external_types_includes(&dir).unwrap();

        assert_eq!(toml.external_tuple_structs, vec!["LocalOnly".to_string(), "Shared1".to_string(), "Shared2".to_string()]);
        assert_eq!(toml.external_const_fns, vec![("Vec3".to_string(), "new".to_string())]);
        // Drained after resolving — calling again is a no-op, not a double-fold.
        assert!(toml.external_types_includes.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_external_types_includes_reports_missing_file_without_exiting() {
        let dir = std::env::temp_dir().join(format!(
            "boring_toml_include_missing_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let mut toml = BoringToml::parse("[external_types]\ninclude = [\"does_not_exist.toml\"]\n");
        let result = toml.resolve_external_types_includes(&dir);
        assert!(result.is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_external_types_includes_does_not_recurse() {
        // An included file's own `include` key is ignored, not followed — deliberately
        // single-level, see `resolve_external_types_includes`'s doc comment.
        let dir = std::env::temp_dir().join(format!(
            "boring_toml_include_no_recurse_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("level1.toml"),
            "[external_types]\ninclude = [\"level2.toml\"]\ntuple_structs = [\"Level1\"]\n",
        ).unwrap();
        std::fs::write(
            dir.join("level2.toml"),
            "[external_types]\ntuple_structs = [\"Level2\"]\n",
        ).unwrap();

        let mut toml = BoringToml::parse("[external_types]\ninclude = [\"level1.toml\"]\n");
        toml.resolve_external_types_includes(&dir).unwrap();

        assert_eq!(toml.external_tuple_structs, vec!["Level1".to_string()]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn derives_section_parses_traits() {
        let src = "[project]\nname = \"demo\"\n\n[derives]\ntraits = [\"Component\", \"Resource\"]\n";
        let toml = BoringToml::parse(src);
        assert_eq!(toml.derive_traits, vec!["Component".to_string(), "Resource".to_string()]);
    }

    #[test]
    fn no_derives_section_is_empty() {
        let src = "[project]\nname = \"demo\"\n";
        let toml = BoringToml::parse(src);
        assert!(toml.derive_traits.is_empty());
    }

    #[test]
    fn derives_include_parses_to_raw_paths() {
        let src = "[derives]\ninclude = [\"../shared/derives.toml\"]\ntraits = [\"LocalOnly\"]\n";
        let toml = BoringToml::parse(src);
        assert_eq!(toml.derive_includes, vec!["../shared/derives.toml".to_string()]);
        assert_eq!(toml.derive_traits, vec!["LocalOnly".to_string()]);
    }

    #[test]
    fn resolve_derive_includes_folds_in_shared_declarations() {
        let dir = std::env::temp_dir().join(format!(
            "boring_toml_derives_include_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("shared_derives.toml"),
            "[derives]\ntraits = [\"Shared1\", \"Shared2\"]\n",
        ).unwrap();

        let mut toml = BoringToml::parse(
            "[derives]\ninclude = [\"shared_derives.toml\"]\ntraits = [\"LocalOnly\"]\n",
        );
        toml.resolve_derive_includes(&dir).unwrap();

        assert_eq!(toml.derive_traits, vec!["LocalOnly".to_string(), "Shared1".to_string(), "Shared2".to_string()]);
        // Drained after resolving — calling again is a no-op, not a double-fold.
        assert!(toml.derive_includes.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_derive_includes_reports_missing_file_without_exiting() {
        let dir = std::env::temp_dir().join(format!(
            "boring_toml_derives_include_missing_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let mut toml = BoringToml::parse("[derives]\ninclude = [\"does_not_exist.toml\"]\n");
        let result = toml.resolve_derive_includes(&dir);
        assert!(result.is_err());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_derive_includes_does_not_recurse() {
        // An included file's own `include` key is ignored, not followed — deliberately
        // single-level, see `resolve_derive_includes`'s doc comment.
        let dir = std::env::temp_dir().join(format!(
            "boring_toml_derives_no_recurse_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("level1.toml"),
            "[derives]\ninclude = [\"level2.toml\"]\ntraits = [\"Level1\"]\n",
        ).unwrap();
        std::fs::write(
            dir.join("level2.toml"),
            "[derives]\ntraits = [\"Level2\"]\n",
        ).unwrap();

        let mut toml = BoringToml::parse("[derives]\ninclude = [\"level1.toml\"]\n");
        toml.resolve_derive_includes(&dir).unwrap();

        assert_eq!(toml.derive_traits, vec!["Level1".to_string()]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn deps_section_parses_plain_path_string() {
        let src = "[deps]\nnumlib = \"../boring-numlib\"\n";
        let toml = BoringToml::parse(src);
        assert_eq!(toml.deps, vec![("numlib".to_string(), "\"../boring-numlib\"".to_string())]);
    }

    #[test]
    fn deps_section_parses_inline_table() {
        let src = "[deps]\nnumlib = { path = \"../boring-numlib\" }\n";
        let toml = BoringToml::parse(src);
        assert_eq!(toml.deps, vec![("numlib".to_string(), "{ path = \"../boring-numlib\" }".to_string())]);
    }

    #[test]
    fn resolve_deps_resolves_path_to_src_dir() {
        let dir = PathBuf::from("/some/project");
        let toml = BoringToml::parse("[deps]\nnumlib = \"../boring-numlib\"\n");
        let resolved = toml.resolve_deps(&dir).unwrap();
        assert_eq!(resolved.get("numlib"), Some(&dir.join("../boring-numlib").join("src")));
    }

    #[test]
    fn resolve_deps_accepts_inline_table_path_form() {
        let dir = PathBuf::from("/some/project");
        let toml = BoringToml::parse("[deps]\nnumlib = { path = \"../boring-numlib\" }\n");
        let resolved = toml.resolve_deps(&dir).unwrap();
        assert_eq!(resolved.get("numlib"), Some(&dir.join("../boring-numlib").join("src")));
    }

    // Real git clone/fetch/checkout coverage (via a local, no-network fixture repo) lives in
    // `git_deps::tests` — these here only cover `parse_dep_value`'s pure parsing of the `git`
    // inline-table shape, no subprocess involved.

    #[test]
    fn parse_dep_value_git_defaults_to_default_ref() {
        let spec = BoringToml::parse_dep_value("{ git = \"https://example.com/x\" }").unwrap();
        assert_eq!(spec, DepSpec::Git { url: "https://example.com/x".to_string(), gitref: GitRef::Default });
    }

    #[test]
    fn parse_dep_value_git_accepts_branch_tag_rev() {
        let branch = BoringToml::parse_dep_value("{ git = \"u\", branch = \"main\" }").unwrap();
        assert_eq!(branch, DepSpec::Git { url: "u".to_string(), gitref: GitRef::Branch("main".to_string()) });

        let tag = BoringToml::parse_dep_value("{ git = \"u\", tag = \"v1.0\" }").unwrap();
        assert_eq!(tag, DepSpec::Git { url: "u".to_string(), gitref: GitRef::Tag("v1.0".to_string()) });

        let rev = BoringToml::parse_dep_value("{ git = \"u\", rev = \"abc123\" }").unwrap();
        assert_eq!(rev, DepSpec::Git { url: "u".to_string(), gitref: GitRef::Rev("abc123".to_string()) });

        // Key order shouldn't matter.
        let reordered = BoringToml::parse_dep_value("{ rev = \"abc123\", git = \"u\" }").unwrap();
        assert_eq!(reordered, DepSpec::Git { url: "u".to_string(), gitref: GitRef::Rev("abc123".to_string()) });
    }

    #[test]
    fn parse_dep_value_rejects_multiple_refs() {
        let err = BoringToml::parse_dep_value("{ git = \"u\", branch = \"main\", tag = \"v1\" }").unwrap_err();
        assert!(err.contains("more than one"), "unexpected error: {err}");
    }

    #[test]
    fn parse_dep_value_rejects_path_and_git_together() {
        let err = BoringToml::parse_dep_value("{ path = \"../x\", git = \"u\" }").unwrap_err();
        assert!(err.contains("both 'path' and 'git'"), "unexpected error: {err}");
    }

    #[test]
    fn resolve_deps_rejects_reserved_names() {
        for reserved in ["std", "crate", "boring"] {
            let src = format!("[deps]\n{reserved} = \"../whatever\"\n");
            let toml = BoringToml::parse(&src);
            let err = toml.resolve_deps(&PathBuf::from("/some/project")).unwrap_err();
            assert!(err.contains("reserved"), "unexpected error for '{reserved}': {err}");
        }
    }

    #[test]
    fn no_deps_section_is_empty() {
        let toml = BoringToml::parse("[project]\nname = \"demo\"\n");
        assert!(toml.deps.is_empty());
        assert!(toml.resolve_deps(&PathBuf::from("/some/project")).unwrap().is_empty());
    }

    #[test]
    fn external_fns_section_parses_method_and_free_function_keys() {
        let src = "[external_fns]\n\"ZipFile::readToEnd\" = [\"&mut\"]\n\"std::mem::swap\" = [\"&mut\", \"&mut\"]\n";
        let toml = BoringToml::parse(src);
        assert_eq!(toml.external_fns, vec![
            ("ZipFile".to_string(), "readToEnd".to_string(), vec!["&mut".to_string()]),
            ("std::mem".to_string(), "swap".to_string(), vec!["&mut".to_string(), "&mut".to_string()]),
        ]);
    }

    #[test]
    fn no_external_fns_section_is_empty() {
        let toml = BoringToml::parse("[project]\nname = \"demo\"\n");
        assert!(toml.external_fns.is_empty());
    }

    #[test]
    fn external_fns_key_without_double_colon_is_skipped() {
        let src = "[external_fns]\n\"NotAPair\" = [\"&mut\"]\n\"Tree::from_str\" = [\"\", \"&\"]\n";
        let toml = BoringToml::parse(src);
        assert_eq!(toml.external_fns, vec![
            ("Tree".to_string(), "from_str".to_string(), vec!["".to_string(), "&".to_string()]),
        ]);
    }

    #[test]
    fn external_fns_key_splits_on_last_double_colon() {
        // A free function's qualifier is a module path that can itself contain further
        // `::` — unlike `[external_types]`'s `const_fns` (first-`::` split, safe there
        // since a type name never contains `::`), this must split on the LAST `::` so
        // `"std::mem::swap"` yields qualifier `"std::mem"`, not `"std"`.
        let src = "[external_fns]\n\"std::mem::swap\" = [\"&mut\", \"&mut\"]\n";
        let toml = BoringToml::parse(src);
        assert_eq!(toml.external_fns, vec![
            ("std::mem".to_string(), "swap".to_string(), vec!["&mut".to_string(), "&mut".to_string()]),
        ]);
    }

    #[test]
    fn external_fns_borrow_array_preserves_positional_empty_string() {
        // Regression test for the exact bug the `resvg::render` real-crate repro found:
        // `extract_array` (used by every other bracketed-array value in this file) drops
        // empty-string elements entirely, which is fine for an unordered name list but
        // corrupts a *positional* borrow-form array — `["&", "", "&mut"]` (by-value
        // argument in the middle) must stay three elements, not collapse to two and
        // silently misalign every later position. `extract_borrow_array` (used only for
        // `[external_fns]` values) must keep the blank entry in place.
        let src = "[external_fns]\n\"resvg::render\" = [\"&\", \"\", \"&mut\"]\n";
        let toml = BoringToml::parse(src);
        assert_eq!(toml.external_fns, vec![
            ("resvg".to_string(), "render".to_string(),
             vec!["&".to_string(), "".to_string(), "&mut".to_string()]),
        ]);
    }

    #[test]
    fn external_fns_include_parses_to_raw_paths() {
        let src = "[external_fns]\ninclude = [\"../shared/external_fns.toml\"]\n\"LocalType::localMethod\" = [\"&mut\"]\n";
        let toml = BoringToml::parse(src);
        assert_eq!(toml.external_fns_includes, vec!["../shared/external_fns.toml".to_string()]);
        assert_eq!(toml.external_fns, vec![
            ("LocalType".to_string(), "localMethod".to_string(), vec!["&mut".to_string()]),
        ]);
    }

    #[test]
    fn resolve_external_fns_includes_folds_in_shared_declarations() {
        let dir = std::env::temp_dir().join(format!("boring_test_external_fns_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shared_path = dir.join("shared_external_fns.toml");
        std::fs::write(&shared_path, "[external_fns]\n\"Shared::sharedMethod\" = [\"&\"]\n").unwrap();
        let src = format!("[external_fns]\ninclude = [\"{}\"]\n\"Local::localMethod\" = [\"&mut\"]\n", shared_path.file_name().unwrap().to_str().unwrap());
        let mut toml = BoringToml::parse(&src);
        toml.resolve_external_fns_includes(&dir).unwrap();
        assert!(toml.external_fns.contains(&("Shared".to_string(), "sharedMethod".to_string(), vec!["&".to_string()])));
        assert!(toml.external_fns.contains(&("Local".to_string(), "localMethod".to_string(), vec!["&mut".to_string()])));
        std::fs::remove_dir_all(&dir).ok();
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
    let mut toml = BoringToml::parse(&src);
    if toml.name.is_empty() {
        eprintln!("error: boring.toml is missing a `name` field");
        process::exit(1);
    }
    // `[external_types]` `include` paths are resolved relative to boring.toml's own
    // directory (CWD here, since `toml_path` is always the bare "boring.toml" above) —
    // not relative to whatever directory `boring build` happens to be run from otherwise.
    let toml_dir = toml_path.parent().unwrap_or_else(|| Path::new("."));
    if let Err(e) = toml.resolve_external_types_includes(toml_dir) {
        eprintln!("error: {}", e);
        process::exit(1);
    }
    if let Err(e) = toml.resolve_derive_includes(toml_dir) {
        eprintln!("error: {}", e);
        process::exit(1);
    }
    if let Err(e) = toml.resolve_external_fns_includes(toml_dir) {
        eprintln!("error: {}", e);
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
    eprintln!("    boring build --target rocm               Emit a ROCm/HIP Cargo project from boring.toml (AMD GPUs)");
    eprintln!("    boring build --target rocm <file.br>     Emit a ROCm/HIP Cargo project from a single file");
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
    // Merge `boring.toml`'s `[external_types]` supplement into the CLI-flag-derived config
    // — same "layer project config on top" shape as `config_with_dir` below in
    // `emit_rust_to_dir`, just at the `boring.toml`-loading point instead of the
    // source-file-loading point.
    let config = transpiler::TranspileConfig {
        external_tuple_structs: toml.external_tuple_structs.clone(),
        external_const_fns: toml.external_const_fns.clone(),
        known_derives: toml.derive_traits.clone(),
        external_fns: toml.external_fns.clone(),
        ..config
    };
    emit_rust_with_version_and_config(&toml.main, &toml.version, config, &toml.dependencies);
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
    let mut target_rocm   = false;
    let mut target_metal  = false;
    let mut target_wgpu   = false;
    let mut mode = TranspileMode::Strict;
    let mut threading = ThreadingMode::Multi;
    let mut inline_auto_bytes: usize = 256;
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
                    Some("rocm")   => target_rocm   = true,
                    Some("metal")  => target_metal  = true,
                    Some("wgpu")   => target_wgpu   = true,
                    Some(t) => {
                        eprintln!("error: unknown target '{}'", t);
                        eprintln!("hint:  supported targets: kernel, cuda, rocm, metal, wgpu");
                        process::exit(1);
                    }
                    None => {
                        eprintln!("error: --target requires a value");
                        eprintln!("hint:  supported targets: kernel, cuda, rocm, metal, wgpu");
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
            "--inline-auto-bytes" => {
                i += 1;
                match build_args.get(i).and_then(|s| s.parse::<usize>().ok()) {
                    Some(n) => inline_auto_bytes = n,
                    None => {
                        eprintln!("error: --inline-auto-bytes requires a positive integer");
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
    if target_rocm && threading != ThreadingMode::Multi {
        eprintln!("error: --threading is not available for the rocm target");
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

    // ROCm target — short-circuit before the general config path.
    if target_rocm {
        match file {
            Some(path) => { emit_rocm(path, "0.1.0"); return; }
            None => {
                let (toml, _) = load_project_toml();
                emit_rocm(&toml.main, &toml.version);
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

    let config = TranspileConfig { mode, threading, inline_auto_bytes, instrument, sanitize, source_dir: PathBuf::new(), gpu_kernels: Vec::new(), is_gpu_target: false, gpu_top_level_handled_by_host: false, external_tuple_structs: Vec::new(), external_const_fns: Vec::new(), known_derives: Vec::new(), deps: std::collections::HashMap::new(), external_fns: Vec::new() };

    if emit_rust {
        match file {
            Some(path) => {
                // Same `[external_types]`/`[derives]`/`external_fns` merge as the `None`
                // branch below — needed here too since `boring build --emit-rust some/file.br`
                // (file passed explicitly) and `boring build --emit-rust` (relying on
                // boring.toml's `main` field pointing at that same file) must produce the
                // same transpile config for the same source file. `find_project_root` is the
                // same helper `print_rust`'s own `[deps]` resolution below already uses to
                // locate this file's `boring.toml`.
                let config = if let Some(root) = find_project_root(Path::new(path)) {
                    match std::fs::read_to_string(root.join("boring.toml")) {
                        Ok(toml_src) => {
                            let mut toml = BoringToml::parse(&toml_src);
                            if let Err(e) = toml.resolve_external_types_includes(&root) {
                                eprintln!("error: {}", e);
                                process::exit(1);
                            }
                            if let Err(e) = toml.resolve_derive_includes(&root) {
                                eprintln!("error: {}", e);
                                process::exit(1);
                            }
                            if let Err(e) = toml.resolve_external_fns_includes(&root) {
                                eprintln!("error: {}", e);
                                process::exit(1);
                            }
                            transpiler::TranspileConfig {
                                external_tuple_structs: toml.external_tuple_structs.clone(),
                                external_const_fns: toml.external_const_fns.clone(),
                                known_derives: toml.derive_traits.clone(),
                                external_fns: toml.external_fns.clone(),
                                ..config
                            }
                        }
                        Err(_) => config,
                    }
                } else {
                    config
                };
                print_rust(path, config)
            }
            None => {
                let (toml, _) = load_project_toml();
                // Same `[external_types]` merge as `build_project_with_config` — needed
                // here too since `boring build --emit-rust` (no file arg) is a real,
                // separate project-mode entry point that also loads `boring.toml` but
                // previously skipped this merge (`--emit-rust` only prints transpiled Rust
                // text, it never touches Cargo.toml, so `toml.dependencies` genuinely
                // doesn't apply here — but `external_tuple_structs`/`external_const_fns`
                // affect the emitted Rust text itself, so they do).
                let config = transpiler::TranspileConfig {
                    external_tuple_structs: toml.external_tuple_structs.clone(),
                    external_const_fns: toml.external_const_fns.clone(),
                    known_derives: toml.derive_traits.clone(),
                    external_fns: toml.external_fns.clone(),
                    ..config
                };
                print_rust(&toml.main, config);
            }
        }
        return;
    }

    let project_dir = match (target_kernel, file, output_dir) {
        (true,  Some(path), _)          => { emit_kernel(path); return; }
        (true,  None, _)                => { build_project_kernel(); return; }
        (false, Some(path), Some(dir))  => { emit_rust_to_dir(path, "0.1.0", config, dir.clone(), &[]); dir }
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
    let program = desugar_labeled_array::desugar_labeled_array(program);

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
        // If that project's boring.toml declares [deps] (named dependencies on other
        // Boring projects — see docs/cross-project-code-sharing-gap.md), resolve them
        // now so `use <name>.xxx` works. Errors here (a reserved name, an unsupported
        // `git = ...` entry, ...) are fatal — same fail-fast convention as
        // `load_project_toml`'s own [external_types]/[derives] include resolution.
        if let Ok(toml_src) = std::fs::read_to_string(root.join("boring.toml")) {
            match BoringToml::parse(&toml_src).resolve_deps(&root) {
                Ok(deps) => interp.set_deps(deps),
                Err(e) => {
                    eprintln!("error: {e}");
                    process::exit(1);
                }
            }
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
    emit_rust_with_version_and_config(path, "0.1.0", config, &[]);
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
    let program = desugar_labeled_array::desugar_labeled_array(program);
    if report_check_result(&path, &source, checker::check(&program)) { process::exit(1); }
    // Same [deps] resolution as emit_rust_to_dir, so `--emit-rust` (project mode or a
    // standalone file) resolves `use <name>.xxx` identically to a real `boring build`.
    let mut deps = config.deps.clone();
    if let Some(root) = find_project_root(&path) {
        if let Ok(toml_src) = std::fs::read_to_string(root.join("boring.toml")) {
            match BoringToml::parse(&toml_src).resolve_deps(&root) {
                Ok(resolved) => deps.extend(resolved),
                Err(e) => {
                    eprintln!("error: {e}");
                    process::exit(1);
                }
            }
        }
    }
    // Same `source_dir` derivation as `emit_rust_to_dir` — without it, a local
    // `use <file>` resolves relative to the *process* cwd instead of the source
    // file's own directory, so `--emit-rust` run from anywhere other than the
    // file's directory would silently fail to find it (falls through to being
    // emitted as a bogus external `use` path instead of inlining the module).
    let source_dir = path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
    let config = transpiler::TranspileConfig { source_dir, deps, ..config };
    let out = transpiler::transpile_with_config(&program, config);
    report_transpile_warnings(&path, &source, &out.warnings);
    if !out.errors.is_empty() { report_transpile_errors(&path, &source, &out.errors); process::exit(1); }
    print!("{}", out.code);
    // `--emit-rust` prints a single Rust stream on stdout, so cross-file/cross-project
    // `use` imports (which the project-mode path below writes as their own `src/<name>.rs`
    // and pulls in via `include!`, see `emit_rust_to_dir`) have nowhere else to go — inline
    // each module's code directly here instead, under a banner naming its source file.
    // Module code never re-emits the prelude/imports (see `prelude_emitted` in
    // `transpiler::mod::emit_program`), so simple concatenation into one flat namespace
    // is exactly what `include!` would have produced anyway.
    for (mod_name, mod_code) in &out.modules {
        println!("\n// ─── module: {} ───", mod_name);
        print!("{}", mod_code);
    }
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

fn emit_rust_with_version_and_config(path: &str, version: &str, config: transpiler::TranspileConfig, extra_deps: &[String]) {
    let path = PathBuf::from(path);

    // Determine output project directory next to the source file
    let stem = path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let base_dir = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let project_dir = base_dir.join(rust_dir_name_full(&stem, &config.threading, &config.mode));

    emit_rust_to_dir(path.to_str().unwrap_or(""), version, config, project_dir, extra_deps);
}

fn emit_rust_to_dir(path: &str, version: &str, config: transpiler::TranspileConfig, project_dir: PathBuf, extra_deps: &[String]) {
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
    let program = desugar_labeled_array::desugar_labeled_array(program);

    if report_check_result(&path, &source, checker::check(&program)) {
        process::exit(1);
    }

    let source_dir = path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf();
    // Resolve boring.toml's [deps] the same way run_file does for `boring run` — via
    // find_project_root, so this works uniformly for both project mode (`boring build`,
    // no file arg) and a standalone file (`boring build path/to/file.br`), covering both
    // without needing separate handling in build_project_with_config/the --emit-rust
    // branch. See docs/cross-project-code-sharing-gap.md.
    let mut deps = config.deps.clone();
    if let Some(root) = find_project_root(&path) {
        if let Ok(toml_src) = std::fs::read_to_string(root.join("boring.toml")) {
            match BoringToml::parse(&toml_src).resolve_deps(&root) {
                Ok(resolved) => deps.extend(resolved),
                Err(e) => {
                    eprintln!("error: {e}");
                    process::exit(1);
                }
            }
        }
    }
    let config_with_dir = transpiler::TranspileConfig { source_dir, deps, ..config.clone() };
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
    // Extra dependencies declared under `[dependencies]` in `boring.toml` (e.g.
    // `bevy = { version = "0.19", default-features = false }`) — copied verbatim,
    // one per line, after the crates the transpiler adds on its own. This is the
    // only source of dependencies Boring doesn't infer from `.br` usage itself.
    let user_deps = if extra_deps.is_empty() {
        String::new()
    } else {
        format!("\n{}\n", extra_deps.join("\n"))
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
tokio = {{ version = "1", features = ["full"] }}{stream_deps}{log_dep}{thiserror_dep}{reqwest_dep}{tokio_util_dep}{serde_dep}{local_channel_dep}{user_deps}"#
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
    let mut deps: std::collections::HashMap<String, PathBuf> = std::collections::HashMap::new();
    // Mirror `run_file`'s project-root discovery: a file under `test/` or
    // `examples/` can `use` a module living in the project's `src/` directory
    // without requiring `BORING_PATH` to be set manually. Also resolve that
    // project's boring.toml [deps] the same way, so `use <name>.xxx` works
    // under GPU targets too — see docs/cross-project-code-sharing-gap.md.
    if let Some(root) = find_project_root(&path) {
        let src_dir = root.join("src");
        if src_dir.is_dir() {
            search_paths.push(src_dir);
        }
        if let Ok(toml_src) = std::fs::read_to_string(root.join("boring.toml")) {
            match BoringToml::parse(&toml_src).resolve_deps(&root) {
                Ok(resolved) => deps = resolved,
                Err(e) => {
                    eprintln!("error: {e}");
                    process::exit(1);
                }
            }
        }
    }
    if let Ok(env_path) = std::env::var("BORING_PATH") {
        search_paths.extend(std::env::split_paths(&env_path));
    }
    merge_into(&path, &mut visited, &mut items, &search_paths, &deps);
    desugar_labeled_array::desugar_labeled_array(ast::Program { items })
}

fn merge_into(
    path: &Path,
    visited: &mut std::collections::HashSet<PathBuf>,
    items: &mut Vec<ast::Item>,
    search_paths: &[PathBuf],
    deps: &std::collections::HashMap<String, PathBuf>,
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
            // `use boring.<module>` — first-party stdlib, resolved against the embedded
            // source in `stdlib_embed` rather than the filesystem. Checked before the
            // on-disk search below since `boring` is never a real sibling file/crate.
            if u.path.first().map(String::as_str) == Some("boring") {
                let module = u.path.get(1).map(String::as_str).unwrap_or("");
                match stdlib_embed::lookup(module) {
                    Some(src) => {
                        merge_stdlib_into(module, src, visited, items, search_paths, deps);
                        continue;
                    }
                    None => {
                        eprintln!("error: unknown boring stdlib module '{}'", module);
                        process::exit(1);
                    }
                }
            }
            // `use <name>.xxx` for a declared [deps] dependency (see BoringToml::resolve_deps,
            // docs/cross-project-code-sharing-gap.md) -- resolves against that dependency's
            // own root instead of the generic search-path scan below. A file genuinely
            // missing under a *declared* dependency is a hard error, unlike an ordinary
            // unresolved `use` (kept as a literal Rust import a few lines down) -- the user
            // named this dependency for exactly this purpose.
            if let Some(root) = u.path.first() {
                if let Some(dep_root) = deps.get(root.as_str()) {
                    let rel: PathBuf = u.path[1..].iter().collect::<PathBuf>().with_extension("br");
                    let candidate = dep_root.join(&rel);
                    if candidate.exists() {
                        merge_into(&candidate, visited, items, search_paths, deps);
                        continue;
                    }
                    eprintln!(
                        "error: cannot find '{}' in dependency '{}' (looked in {})",
                        rel.display(), root, dep_root.display()
                    );
                    process::exit(1);
                }
            }
            let rel: PathBuf = u.path.iter().collect::<PathBuf>().with_extension("br");
            let candidate = std::iter::once(&dir)
                .chain(search_paths.iter())
                .map(|base| base.join(&rel))
                .find(|c| c.exists());
            if let Some(candidate) = candidate {
                merge_into(&candidate, visited, items, search_paths, deps);
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

/// `use boring.<module>` counterpart of `merge_into` for GPU targets --
/// parses embedded stdlib source (already looked up by the caller) instead
/// of reading a file, and recurses into its own `use` items the same way
/// (so a stdlib module can itself `use boring.other` or a sibling file).
fn merge_stdlib_into(
    module: &str,
    source: &str,
    visited: &mut std::collections::HashSet<PathBuf>,
    items: &mut Vec<ast::Item>,
    search_paths: &[PathBuf],
    deps: &std::collections::HashMap<String, PathBuf>,
) {
    let synthetic = stdlib_embed::synthetic_path(module);
    if !visited.insert(synthetic) {
        return; // already merged (circular or duplicate `use`)
    }

    let tokens = match lexer::lex_all(source) {
        Ok(t) => t,
        Err(errors) => {
            report_lex_errors(Path::new(&format!("boring.{module}")), source, &errors);
            process::exit(1);
        }
    };
    let program = match parser::parse(tokens) {
        Ok(p) => p,
        Err(e) => {
            report_error(Path::new(&format!("boring.{module}")), source, e.line(), e.col(), e.len(), &e.msg());
            process::exit(1);
        }
    };

    for item in program.items {
        if let ast::Item::Use(u) = &item {
            if u.path.first().map(String::as_str) == Some("boring") {
                let sub_module = u.path.get(1).map(String::as_str).unwrap_or("");
                match stdlib_embed::lookup(sub_module) {
                    Some(src) => {
                        merge_stdlib_into(sub_module, src, visited, items, search_paths, deps);
                        continue;
                    }
                    None => {
                        eprintln!("error: unknown boring stdlib module '{}'", sub_module);
                        process::exit(1);
                    }
                }
            }
            if let Some(root) = u.path.first() {
                if let Some(dep_root) = deps.get(root.as_str()) {
                    let rel: PathBuf = u.path[1..].iter().collect::<PathBuf>().with_extension("br");
                    let candidate = dep_root.join(&rel);
                    if candidate.exists() {
                        merge_into(&candidate, visited, items, search_paths, deps);
                        continue;
                    }
                    eprintln!(
                        "error: cannot find '{}' in dependency '{}' (looked in {})",
                        rel.display(), root, dep_root.display()
                    );
                    process::exit(1);
                }
            }
            let rel: PathBuf = u.path.iter().collect::<PathBuf>().with_extension("br");
            let candidate = search_paths.iter().map(|base| base.join(&rel)).find(|c| c.exists());
            if let Some(candidate) = candidate {
                merge_into(&candidate, visited, items, search_paths, deps);
                continue;
            }
        }
        items.push(item);
    }
}

fn emit_cuda(path: &str, version: &str) {
    let program = parse_and_merge_program(path);
    let path = PathBuf::from(path);

    // Kernel-dispatch-qualifier check only -- the four GPU emit_* functions used
    // to skip the checker entirely (only `run`/plain `build` called it), so e.g. a
    // `'shared`-qualified kernel instance dispatched via `kernel:` went unrejected
    // on every GPU target. The FULL checker isn't safe to turn on here yet (a
    // pre-existing, unrelated GPU-resident-tuple-return opacity false positive --
    // see `checker::check_kernel_dispatch_only`'s doc), so only this one check runs.
    let source = std::fs::read_to_string(&path).unwrap_or_default();
    if report_check_result(&path, &source, checker::check_kernel_dispatch_only(&program)) {
        process::exit(1);
    }

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

fn emit_rocm(path: &str, version: &str) {
    let program = parse_and_merge_program(path);
    let path = PathBuf::from(path);

    // See `emit_cuda`'s identical check for why this is needed here too.
    let source = std::fs::read_to_string(&path).unwrap_or_default();
    if report_check_result(&path, &source, checker::check_kernel_dispatch_only(&program)) {
        process::exit(1);
    }

    let stem = path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_string());
    let base_dir    = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let project_dir = base_dir.join(format!("{}_rocm", stem));

    let rocm_out = transpiler::rocm::transpile_rocm(&program, &stem, version);
    if !rocm_out.errors.is_empty() {
        let source = std::fs::read_to_string(&path).unwrap_or_default();
        report_transpile_errors(&path, &source, &rocm_out.errors);
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
    if let Err(e) = std::fs::write(&main_rs, &rocm_out.host_rs) {
        eprintln!("error: cannot write '{}': {}", main_rs.display(), e);
        process::exit(1);
    }

    // kernels/main.hip
    let hip_file = kernels_dir.join("main.hip");
    if let Err(e) = std::fs::write(&hip_file, &rocm_out.device_hip) {
        eprintln!("error: cannot write '{}': {}", hip_file.display(), e);
        process::exit(1);
    }

    // build.rs
    let build_rs = project_dir.join("build.rs");
    if let Err(e) = std::fs::write(&build_rs, &rocm_out.build_rs) {
        eprintln!("error: cannot write '{}': {}", build_rs.display(), e);
        process::exit(1);
    }

    // Cargo.toml
    let cargo_toml = project_dir.join("Cargo.toml");
    if let Err(e) = std::fs::write(&cargo_toml, &rocm_out.cargo_toml) {
        eprintln!("error: cannot write '{}': {}", cargo_toml.display(), e);
        process::exit(1);
    }

    eprintln!("Generated ROCm project at '{}'", project_dir.display());
    eprintln!("  Requires the ROCm toolkit (hipcc) and an AMD GPU.");
    eprintln!("  cd {} && cargo build", project_dir.display());
    if !rocm_out.kernel_names.is_empty() {
        eprintln!("  Kernels: {}", rocm_out.kernel_names.join(", "));
    }
}

fn emit_metal(path: &str, version: &str) {
    let program = parse_and_merge_program(path);
    let path = PathBuf::from(path);

    // See `emit_cuda`'s identical check for why this is needed here too.
    let source = std::fs::read_to_string(&path).unwrap_or_default();
    if report_check_result(&path, &source, checker::check_kernel_dispatch_only(&program)) {
        process::exit(1);
    }

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

    // See `emit_cuda`'s identical check for why this is needed here too.
    let source = std::fs::read_to_string(&path).unwrap_or_default();
    if report_check_result(&path, &source, checker::check_kernel_dispatch_only(&program)) {
        process::exit(1);
    }

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

    // shaders/main_emulated.wgsl — only emitted when some kernel uses `gpu.warp.*`;
    // the host chooses between this and main.wgsl at runtime based on whether the
    // adapter has `wgpu::Features::SUBGROUP` (see transpiler::wgpu::device::WarpMode).
    if let Some(emulated) = &wgpu_out.device_wgsl_emulated {
        let wgsl_emulated_file = shaders_dir.join("main_emulated.wgsl");
        if let Err(e) = std::fs::write(&wgsl_emulated_file, emulated) {
            eprintln!("error: cannot write '{}': {}", wgsl_emulated_file.display(), e);
            process::exit(1);
        }
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
    let program = desugar_labeled_array::desugar_labeled_array(program);

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


