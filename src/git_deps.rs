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

//! Resolves a `boring.toml` `[deps]` `git = "..."` dependency by shelling out to the `git`
//! CLI — no new Cargo dependency (this project has exactly two: `thiserror`/`rayon`), matching
//! the rest of the compiler's "hand-rolled, minimal" style (e.g. `BoringToml`'s own tiny
//! hand-written TOML-subset parser). See `docs/cross-project-code-sharing-gap.md` and
//! `docs/book.md` §15 for the user-facing feature.
//!
//! Clones are cached in a persistent directory (see `cache_root`), keyed by `(url, gitref)` —
//! deliberately one full clone per distinct ref of a repo rather than a shared bare-repo +
//! `git worktree` layout: simpler to implement and reason about, at the cost of some
//! duplicated disk space if a project depends on the same repo at two different refs (expected
//! to be rare). A dependency directory, once cloned, is only ever touched again to *refresh*
//! it (fetch + checkout) — never deleted and re-cloned from scratch — so resolving the same
//! dependency twice needs no network at all once a `rev`-pinned dependency's commit is already
//! present, and only a small fetch for a `branch`/`tag`/default dependency.

use crate::GitRef;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Root directory under which every git dependency gets its own subdirectory. Checked in this
/// order: `BORING_CACHE_DIR` env var (mainly for tests — keeps them from touching or racing on
/// the real user cache — but also a legitimate way for a user/CI job to redirect the cache);
/// `XDG_CACHE_HOME` (honored on any OS that sets it, not just Linux); a platform-conventional
/// default derived from `HOME`; `std::env::temp_dir()` as an unlikely last resort if even
/// `HOME` isn't set. No `dirs`/`directories` crate — this project has none of the "look up the
/// OS-conventional cache dir" infrastructure today, and pulling one in for a single call site
/// isn't worth a new dependency.
fn cache_root() -> PathBuf {
    if let Ok(dir) = std::env::var("BORING_CACHE_DIR") {
        return PathBuf::from(dir).join("git-deps");
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("boring").join("git-deps");
    }
    if let Ok(home) = std::env::var("HOME") {
        let home = PathBuf::from(home);
        if cfg!(target_os = "macos") {
            return home.join("Library/Caches/boring/git-deps");
        }
        return home.join(".cache/boring/git-deps");
    }
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        return PathBuf::from(local).join("boring").join("git-deps");
    }
    std::env::temp_dir().join("boring").join("git-deps")
}

/// A small, non-cryptographic hash for naming a cache directory. Only needs to be unique
/// enough to avoid accidental collisions on one machine's local cache — not stable across Rust
/// versions or suitable for anything security-sensitive. `std::collections::hash_map::
/// DefaultHasher` (stdlib, no new dependency) is exactly the right tool for that bar.
fn short_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// The cache directory for one specific `(url, gitref)` pair. Named `<slug>-<hash>` — the slug
/// (last non-empty path segment of the URL, sanitized) is purely for human debuggability when
/// browsing the cache directory by hand; the hash is what actually guarantees uniqueness (the
/// slug alone would collide for e.g. two different hosts' `.../foo` repos, or the same repo at
/// two different refs).
fn dep_cache_dir(url: &str, gitref: &GitRef) -> PathBuf {
    let slug: String = url
        .trim_end_matches('/')
        .trim_end_matches(".git")
        .rsplit('/')
        .next()
        .unwrap_or("dep")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let key = format!("{url}#{gitref:?}");
    cache_root().join(format!("{}-{:016x}", slug, short_hash(&key)))
}

/// Runs `git` with the given args, in `dir` if given (via `-C`), returning stdout (trimmed) on
/// success. `context` names the operation for the error message (e.g. `"clone"`, `"fetch"`) —
/// distinguishes a spawn failure (git itself missing from PATH) from git exiting non-zero
/// (bad URL, auth failure, unknown ref, ...), mirroring `run_cargo_build`'s (`src/main.rs`)
/// existing spawn-vs-exit-code error style for subprocess calls.
fn run_git(dir: Option<&Path>, args: &[&str], context: &str) -> Result<String, String> {
    let mut cmd = Command::new("git");
    if let Some(d) = dir {
        cmd.arg("-C").arg(d);
    }
    cmd.args(args);
    let output = cmd.output().map_err(|e| {
        format!("failed to run 'git {}' ({context}): {e} — is git installed and on PATH?", args.join(" "))
    })?;
    if !output.status.success() {
        return Err(format!(
            "'git {}' ({context}) failed:\n{}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// True if `rev` names a commit object already present in the repo at `dir` — lets a
/// `rev`-pinned dependency that's already fully cached resolve with zero network access.
fn commit_present(dir: &Path, rev: &str) -> bool {
    Command::new("git")
        .arg("-C").arg(dir)
        .args(["cat-file", "-e", &format!("{rev}^{{commit}}")])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Resolves a `{ git = "<url>", ... }` `[deps]` entry to its `src/` directory, cloning into (or
/// refreshing within) this dependency's persistent cache directory (`dep_cache_dir`). Called
/// from `BoringToml::resolve_deps`, which prefixes any error here with the dependency's name.
pub(crate) fn resolve_git_dep(url: &str, gitref: &GitRef) -> Result<PathBuf, String> {
    let dir = dep_cache_dir(url, gitref);
    if dir.join(".git").is_dir() {
        refresh(&dir, gitref)?;
    } else {
        std::fs::create_dir_all(dir.parent().unwrap_or(Path::new("."))).map_err(|e| {
            format!("cannot create git dependency cache directory '{}': {e}", dir.display())
        })?;
        clone_fresh(url, &dir, gitref)?;
    }
    let src = dir.join("src");
    if !src.is_dir() {
        return Err(format!(
            "git repository '{url}' has no 'src/' directory (looked in {}) — a [deps] git \
             dependency needs the same src/ convention as a path dependency",
            src.display()
        ));
    }
    Ok(src)
}

fn clone_fresh(url: &str, dir: &Path, gitref: &GitRef) -> Result<(), String> {
    match gitref {
        // An arbitrary commit may not be the tip of any ref a shallow clone would fetch, so
        // this needs full history up front.
        GitRef::Rev(rev) => {
            run_git(None, &["clone", url, &dir.to_string_lossy()], "clone")?;
            run_git(Some(dir), &["checkout", rev], "checkout")?;
        }
        GitRef::Branch(b) => {
            run_git(None, &["clone", "--depth", "1", "--branch", b, url, &dir.to_string_lossy()], "clone")?;
        }
        GitRef::Tag(t) => {
            run_git(None, &["clone", "--depth", "1", "--branch", t, url, &dir.to_string_lossy()], "clone")?;
        }
        GitRef::Default => {
            run_git(None, &["clone", "--depth", "1", url, &dir.to_string_lossy()], "clone")?;
        }
    }
    Ok(())
}

/// Updates an already-cloned dependency in place. `Rev` needs no network at all once the
/// commit is present; `Branch`/`Tag`/`Default` always attempt a fetch (they name a *mutable*
/// ref, so staleness is a real concern) but fall back to what's already checked out if the
/// fetch fails and something is already usable — a dependency shouldn't become hard-broken
/// just because the network happens to be down for one build.
fn refresh(dir: &Path, gitref: &GitRef) -> Result<(), String> {
    match gitref {
        GitRef::Rev(rev) => {
            if commit_present(dir, rev) {
                return run_git(Some(dir), &["checkout", rev], "checkout").map(|_| ());
            }
            // Not present yet -- try fetching it directly (works on hosts that allow fetching
            // an arbitrary reachable sha, e.g. GitHub); if the host rejects that, deepen the
            // existing shallow-or-partial history and retry.
            if run_git(Some(dir), &["fetch", "origin", rev], "fetch").is_err() {
                run_git(Some(dir), &["fetch", "--unshallow", "origin"], "fetch --unshallow")
                    .or_else(|_| run_git(Some(dir), &["fetch", "--all"], "fetch --all"))?;
            }
            run_git(Some(dir), &["checkout", rev], "checkout").map(|_| ())
        }
        GitRef::Branch(b) => refresh_mutable_ref(dir, Some(b)),
        GitRef::Tag(t) => refresh_mutable_ref(dir, Some(t)),
        GitRef::Default => refresh_mutable_ref(dir, None),
    }
}

fn refresh_mutable_ref(dir: &Path, ref_name: Option<&str>) -> Result<(), String> {
    let mut fetch_args = vec!["fetch", "--depth", "1", "origin"];
    if let Some(r) = ref_name {
        fetch_args.push(r);
    }
    match run_git(Some(dir), &fetch_args, "fetch") {
        Ok(_) => run_git(Some(dir), &["checkout", "FETCH_HEAD"], "checkout").map(|_| ()),
        Err(e) => {
            // Offline (or the remote is unreachable) -- non-fatal as long as something is
            // already checked out from a previous successful resolution.
            if run_git(Some(dir), &["rev-parse", "--verify", "-q", "HEAD"], "rev-parse").is_ok() {
                eprintln!("warning: could not refresh git dependency ({e}); using cached copy");
                Ok(())
            } else {
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `BORING_CACHE_DIR` is a process-wide env var and `cargo test` runs tests in this binary
    // concurrently by default -- every test below that sets it must hold this lock for its
    // whole duration (see each test's own `let _guard = ...`), or two such tests interleaving
    // would silently point `cache_root()` at whichever one last called `set_var`. Combining
    // scenarios into one `#[test]` function where possible (see `resolve_git_dep_clones_and_
    // reuses_cache`) avoids needing this for everything, but a second test (missing src/ dir)
    // still needs its own env-var window, hence the shared lock.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn run(dir: &Path, args: &[&str]) {
        let status = Command::new("git").arg("-C").arg(dir).args(args).status()
            .unwrap_or_else(|e| panic!("failed to run git {:?} in {}: {e}", args, dir.display()));
        assert!(status.success(), "git {:?} failed in {}", args, dir.display());
    }

    /// Sets up a local git repo (no network) with one commit under `src/`, returning its full
    /// commit sha. Used as the "remote" for `resolve_git_dep` tests — git treats a local
    /// filesystem path as a valid remote natively, no `file://` prefix needed.
    fn make_fixture_repo(dir: &Path) -> String {
        std::fs::create_dir_all(dir.join("src")).unwrap();
        run(dir, &["init", "-q", "-b", "main"]);
        run(dir, &["config", "user.email", "test@example.com"]);
        run(dir, &["config", "user.name", "Test"]);
        std::fs::write(dir.join("src").join("lib.br"), "pub struct Marker:\n    pub int value\n").unwrap();
        run(dir, &["add", "."]);
        run(dir, &["commit", "-q", "-m", "initial"]);
        let out = Command::new("git").arg("-C").arg(dir).args(["rev-parse", "HEAD"]).output().unwrap();
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    }

    // One combined test (not several parallel #[test] fns) deliberately: BORING_CACHE_DIR is a
    // process-wide env var, and `cargo test` runs tests concurrently by default -- splitting
    // this into multiple #[test] functions that each set/unset it would race. Exercises: a
    // default-branch clone, a rev-pinned clone, and that resolving the *same* dependency twice
    // reuses the cache (no error, same path) without needing the fixture repo to change.
    #[test]
    fn resolve_git_dep_clones_and_reuses_cache() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = std::env::temp_dir().join(format!("boring_git_dep_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let repo = tmp.join("repo");
        let sha = make_fixture_repo(&repo);
        let repo_url = repo.to_string_lossy().to_string();

        let cache = tmp.join("cache");
        std::env::set_var("BORING_CACHE_DIR", &cache);

        // Default branch.
        let src1 = resolve_git_dep(&repo_url, &GitRef::Default).expect("default-branch clone");
        assert!(src1.join("lib.br").is_file());

        // Resolving again must succeed without the fixture doing anything further -- proves
        // the cache directory is actually being read on the second call.
        let src1_again = resolve_git_dep(&repo_url, &GitRef::Default).expect("cached re-resolve");
        assert_eq!(src1, src1_again);

        // A specific rev, in a *separate* cache directory from the default-branch one (per
        // dep_cache_dir's one-directory-per-(url,ref) design).
        let src2 = resolve_git_dep(&repo_url, &GitRef::Rev(sha.clone())).expect("rev-pinned clone");
        assert!(src2.join("lib.br").is_file());
        assert_ne!(src1, src2);

        // Re-resolving the same rev must need no network at all (commit_present short-circuits
        // straight to checkout) -- if this were broken it would try to fetch from `repo_url`
        // and still succeed here since the fixture still exists, so this mainly guards against
        // a panic/error in that code path, not staleness; the important behavioral guarantee
        // (no fetch attempted) is documented in `refresh`'s doc comment.
        let src2_again = resolve_git_dep(&repo_url, &GitRef::Rev(sha)).expect("cached rev re-resolve");
        assert_eq!(src2, src2_again);

        std::env::remove_var("BORING_CACHE_DIR");
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn resolve_git_dep_reports_missing_src_dir() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = std::env::temp_dir().join(format!("boring_git_dep_test_nosrc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let repo = tmp.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        run(&repo, &["init", "-q", "-b", "main"]);
        run(&repo, &["config", "user.email", "test@example.com"]);
        run(&repo, &["config", "user.name", "Test"]);
        std::fs::write(repo.join("README.md"), "no src/ here\n").unwrap();
        run(&repo, &["add", "."]);
        run(&repo, &["commit", "-q", "-m", "initial"]);

        let cache = tmp.join("cache");
        std::env::set_var("BORING_CACHE_DIR", &cache);
        let err = resolve_git_dep(&repo.to_string_lossy(), &GitRef::Default).unwrap_err();
        assert!(err.contains("no 'src/' directory"), "unexpected error: {err}");
        std::env::remove_var("BORING_CACHE_DIR");

        std::fs::remove_dir_all(&tmp).ok();
    }
}
