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
}

impl SourceError {
    pub fn new(message: impl Into<String>, line: usize, col: usize, len: usize) -> Self {
        SourceError { message: message.into(), line, col, len }
    }

    /// Column/span unknown — e.g. an error attributed to a whole line/statement
    /// rather than a specific token.
    pub fn at_line(message: impl Into<String>, line: usize) -> Self {
        SourceError { message: message.into(), line, col: 0, len: 0 }
    }

    /// Column known, span not tracked — underlines a single character.
    pub fn at(message: impl Into<String>, line: usize, col: usize) -> Self {
        SourceError { message: message.into(), line, col, len: 1 }
    }
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}
