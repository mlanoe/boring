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

//! Embeds the first-party `boring.*` standard library (`stdlib/*.br`)
//! directly into the compiler binary via `include_str!`, mirroring how
//! `interpreter::gpu_profile` already embeds the built-in GPU profile TOMLs.
//!
//! This is necessary because `boring` is only ever distributed as a single
//! binary (`cargo install --git ...` / `cargo install --path .` — see
//! `README.md`); there is no packaging step that could ship a stdlib
//! directory alongside it, and no existing mechanism resolves paths
//! relative to the installed binary. Embedding the source text means
//! `use boring.<module>` works everywhere the `boring` binary runs, with
//! zero setup.
//!
//! `use boring.<module>` is intercepted before the normal `use` resolution
//! (which searches project-relative directories / `BORING_PATH`) in three
//! independent places that each need it:
//!   - the interpreter (`boring run`) — `interpreter::exec_use_decl`
//!   - the transpiler (`boring build`, std/tokio target) — `transpiler::emit_top::emit_use`/`deep_pre_scan`
//!   - the GPU-target CLI merge path (`boring build --target wgpu/cuda/metal`) — `main::merge_into`
//!
//! See `stdlib/README.md` for how to add a new module.

const STD_COLLECTIONS: &str = include_str!("../stdlib/collections.br");

/// Look up the embedded source for `use boring.<module>`.
///
/// Returns `None` if `module` isn't a recognized stdlib module name — every
/// call site treats that as a hard error (an unrecognized `boring.*` name is
/// unambiguously a mistake), unlike the generic filesystem `use` loader,
/// which silently no-ops when a file isn't found (it assumes a native Rust
/// module in that case — not applicable here, `boring` is not a real crate).
pub fn lookup(module: &str) -> Option<&'static str> {
    Some(match module {
        "collections" => STD_COLLECTIONS,
        _ => return None,
    })
}

/// Synthetic path used to key the circular/duplicate-import guards
/// (`Interpreter::loaded`, `Transpiler::loaded`, `merge_into`'s `visited`
/// set) for an embedded stdlib module — these guards are `HashSet<PathBuf>`
/// keyed on real, canonicalized filesystem paths elsewhere; a stdlib module
/// has no file on disk, so it gets a synthetic key instead. The `<...>`
/// bracketing keeps it visually distinct from a real path and makes
/// collision with an actual project file (which could never contain `<`/`>`
/// on any of the platforms `boring` targets) effectively impossible.
pub fn synthetic_path(module: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(format!("<boring-stdlib>/{module}.br"))
}
