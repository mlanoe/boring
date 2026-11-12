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

// Shared "positioned error" payload.
//
// `checker::CheckError`/`CheckWarning`, `transpiler::TranspileError`, and
// `interpreter::RuntimeError` used to be four separately-defined structs with
// the same `{ message, line, col }` shape (`RuntimeError` alone also tracked
// `len`, the underline width `main.rs`'s error printer wants -- the other three
// just hardcoded `1` at the print site instead). Each phase still reports
// through its own name (a `Vec<CheckError>` reads as "checker phase" at a
// glance) -- they're just aliases of this one type now, so there's one
// definition, one `Display` impl, and `len` is available everywhere instead of
// only where `RuntimeError` happened to track it.

#[derive(Debug, Clone)]
pub struct SourceError {
    pub message: String,
    pub line: usize,
    pub col: usize,
    /// Underline width for the caret printed under the offending source text
    /// (see `main.rs`'s `report_error`/`report_warning`). `1` is a reasonable
    /// default for a call site that doesn't track the real span.
    pub len: usize,
    /// Source file the error occurred in when it is *not* the entry file (an imported
    /// module's function/top-level raised it). `None` = the file being run/compiled.
    pub file: Option<std::path::PathBuf>,
}

impl SourceError {
    pub fn new(message: impl Into<String>, line: usize, col: usize, len: usize) -> Self {
        SourceError { message: message.into(), line, col, len, file: None }
    }

    /// Column/span unknown — e.g. an error attributed to a whole line/statement
    /// rather than a specific token.
    pub fn at_line(message: impl Into<String>, line: usize) -> Self {
        SourceError { message: message.into(), line, col: 0, len: 0, file: None }
    }

    /// Column known, span not tracked — underlines a single character.
    pub fn at(message: impl Into<String>, line: usize, col: usize) -> Self {
        SourceError { message: message.into(), line, col, len: 1, file: None }
    }
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

// ---- File identity of merged sibling files -------------------------------------------------
//
// GPU targets merge a project's bare-`use` sibling files (and `[deps]` / stdlib modules) into one
// flat item list, and every later pass (checker, transpiler, desugars) reports a bare
// `line`/`col`. To keep the file identity, the tokens of the Nth merged file (N >= 1) get their
// line shifted by `N << MERGED_LINE_SHIFT`; the entry file (N = 0) is left as is. The diagnostic
// printers (`main.rs`) decode it back (`resolve_merged_line`), so an error is shown against the file it is
// really in, with that file's own line and excerpt.
pub const MERGED_LINE_SHIFT: u32 = 20;
pub const MERGED_LINE_MASK: usize = (1 << MERGED_LINE_SHIFT) - 1;

/// Merged files in merge order (index = line-offset group); `None` source = read from disk.
static MERGED_FILES: std::sync::Mutex<Vec<(std::path::PathBuf, Option<String>)>> = std::sync::Mutex::new(Vec::new());

/// Registers a merged file (once per path) and returns the line offset to add to each of its tokens.
pub fn register_merged_file(path: &std::path::Path, source: Option<&str>) -> usize {
    let mut files = MERGED_FILES.lock().unwrap();
    // Slot 0 is the entry file (never shifted); the entry is not necessarily registered itself.
    if files.is_empty() {
        files.push((std::path::PathBuf::new(), None));
    }
    // Idempotent: a file read by several passes (pre-scan, emission) keeps one identity.
    if let Some(i) = files.iter().position(|(p, _)| p == path) {
        return i << MERGED_LINE_SHIFT;
    }
    files.push((path.to_path_buf(), source.map(str::to_string)));
    (files.len() - 1) << MERGED_LINE_SHIFT
}

/// Decodes a possibly shifted line: the file (and its source) it belongs to when that is not the
/// file the caller passed, and the plain line.
pub fn resolve_merged_line(path: &std::path::Path, source: &str, line: usize) -> (std::path::PathBuf, String, usize) {
    let idx = line >> MERGED_LINE_SHIFT;
    if idx == 0 {
        return (path.to_path_buf(), source.to_string(), line);
    }
    let files = MERGED_FILES.lock().unwrap();
    match files.get(idx) {
        Some((file, embedded)) => {
            let src = match embedded {
                Some(s) => s.clone(),
                None => std::fs::read_to_string(file).unwrap_or_default(),
            };
            (file.clone(), src, line & MERGED_LINE_MASK)
        }
        None => (path.to_path_buf(), source.to_string(), line & MERGED_LINE_MASK),
    }
}

pub fn shift_token_lines(tokens: &mut [crate::lexer::Token], offset: usize) {
    if offset != 0 {
        for t in tokens.iter_mut() {
            if t.line > 0 { t.line += offset; }
        }
    }
}
