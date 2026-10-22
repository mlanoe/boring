// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Shared helper for tests that build+run a generated Cargo project.
//
// Every such project used to get its own `target/` dir next to its `Cargo.toml`
// (100-440MB each for wgpu/screen fixtures, ~7GB under `target/tmp` in total). They now
// all share one build directory per test *binary*, so wgpu/winit/etc. compile once.
//
// Generated projects are all named `test`, so they would overwrite each other's
// executable in the shared dir: the whole cargo invocation (build + run) is serialized
// by `CARGO_LOCK`, exactly as `tests/transpile.rs` does for the same reason.
// `BORING_KEEP_GENERATED=1` is not needed here: the per-test project dirs are a few KB.
#![allow(dead_code)]

use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::Mutex;

static CARGO_LOCK: Mutex<()> = Mutex::new(());

/// Drop-in for the `Command::new("cargo")...output()` chains these tests used.
pub struct SharedCargo(Command, Option<PathBuf>);

impl SharedCargo {
    /// `name` picks the shared directory (use the test file's name).
    pub fn new(name: &str) -> Self {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("shared_target").join(name);
        let mut c = Command::new("cargo");
        c.env("CARGO_TARGET_DIR", dir)
            // Throwaway projects: incremental artifacts are pure disk cost.
            .env("CARGO_INCREMENTAL", "0");
        SharedCargo(c, None)
    }
    pub fn args<I: IntoIterator<Item = S>, S: AsRef<OsStr>>(mut self, a: I) -> Self {
        self.0.args(a);
        self
    }
    pub fn arg<S: AsRef<OsStr>>(mut self, a: S) -> Self {
        // Remember the manifest (the `--manifest-path <p>` pair) so `output` can rename its package.
        if self.1.is_none() && self.0.get_args().last() == Some(OsStr::new("--manifest-path")) {
            self.1 = Some(PathBuf::from(a.as_ref()));
        }
        self.0.arg(a);
        self
    }
    pub fn env<K: AsRef<OsStr>, V: AsRef<OsStr>>(mut self, k: K, v: V) -> Self {
        self.0.env(k, v);
        self
    }
    pub fn output(mut self) -> std::io::Result<Output> {
        if let Some(m) = &self.1 {
            // Generated projects are all named `test`; in a shared target dir they would
            // run each other's executable. Give each one a name unique to its directory.
            if let Ok(t) = std::fs::read_to_string(m) {
                let id: String = m.parent().and_then(|d| d.parent()).and_then(|d| d.file_name())
                    .map(|n| n.to_string_lossy().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect())
                    .unwrap_or_default();
                let t = t.replacen("name = \"test\"", &format!("name = \"t_{id}\""), 1);
                let _ = std::fs::write(m, t);
            }
        }
        let _g = CARGO_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        self.0.output()
    }
}
