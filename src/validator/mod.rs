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

//! Validation passes for the Boring AST.
//!
//! Each pass analyses the AST and produces a list of [`KernelDiagnostic`] items.
//! The caller decides whether to abort (on errors) or continue (on warnings only).

pub mod kernel;

/// Severity level of a kernel validation diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub enum DiagLevel {
    Error,
    Warning,
}

/// A single diagnostic produced by a validation pass.
#[derive(Debug, Clone)]
pub struct KernelDiagnostic {
    pub level:   DiagLevel,
    pub line:    usize,
    pub message: String,
}

/// Run the kernel validation pass over `program` and return all diagnostics.
///
/// The returned list may contain both errors and warnings.  The caller should
/// inspect `diag.level` and abort if any `DiagLevel::Error` is present.
pub fn validate_kernel(program: &crate::ast::Program) -> Vec<KernelDiagnostic> {
    kernel::KernelValidator::new().run(program)
}
