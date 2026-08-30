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
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Network/lock policy for one resolution — `boring run --locked`/`--offline` and `boring build
/// --locked`/`--offline` (see `docs/book.md` §15) thread this down to every place a network or
/// lock-write decision actually gets made, instead of each intermediate function needing two
/// separate bool parameters. Defaults (`locked: false, offline: false`) reproduce today's
/// pre-flag behavior exactly — a resolution with no explicit policy is unaffected.
/// `boring update` never uses this (forcing a fresh resolution past the lock is its entire
/// purpose) — `update_git_dep` always resolves with `DepPolicy::default()` internally.
#[derive(Clone, Copy, Default)]
pub(crate) struct DepPolicy {
    /// Refuse to create a *new* or *changed* `boring.lock` entry — see `resolve_git_dep_with_lock`.
    pub(crate) locked: bool,
    /// Refuse any network access — see `clone_fresh`/`refresh`/`refresh_mutable_ref`.
    pub(crate) offline: bool,
}

impl DepPolicy {
    /// Reads `BORING_LOCKED`/`BORING_OFFLINE` (set, not unset — an *empty* value still counts,
    /// same as every other presence-checked env var in this codebase, e.g. `BORING_PATH`'s
    /// sibling checks) — these are set once by `parse_run_flags`/`parse_build_command` right
    /// after parsing `--locked`/`--offline`, not meant to be set by hand.
    pub(crate) fn from_env() -> Self {
        Self {
            locked: std::env::var("BORING_LOCKED").is_ok(),
            offline: std::env::var("BORING_OFFLINE").is_ok(),
        }
    }
}

impl GitRef {
    /// Stable string form of a *mutable* ref, used as `boring.lock`'s `requested` field so a
    /// later resolution can tell whether `boring.toml` still asks for the same thing (reuse the
    /// lock) or has changed (e.g. switched branches — the stale entry must not be reused).
    /// Never called for `Rev`: an exact commit is already pinned by the user directly in
    /// `boring.toml`, so it never touches the lock at all (see `resolve_git_dep_with_lock`).
    fn to_lock_string(&self) -> String {
        match self {
            GitRef::Default => "default".to_string(),
            GitRef::Branch(b) => format!("branch:{b}"),
            GitRef::Tag(t) => format!("tag:{t}"),
            GitRef::Rev(_) => unreachable!("Rev dependencies never consult the lock"),
        }
    }
}

/// One resolved `boring.lock` entry — see `BoringLock`'s doc comment.
#[derive(Debug, Clone, PartialEq)]
struct LockEntry {
    url: String,
    requested: String,
    resolved: String,
}

/// `boring.lock` — pins each *mutable-ref* git dependency (`branch`/`tag`/the default branch)
/// to the exact commit it last resolved to, so it stops silently drifting between builds or
/// machines the way a plain `git clone --branch main` would. A `rev`-pinned dependency is
/// already exact by construction in `boring.toml` itself and never gets an entry here.
///
/// File format (next to `boring.toml`, same "tiny hand-rolled, no real parser" style as that
/// file itself — flat `<name>.<field> = "value"` lines rather than TOML array-of-tables, which
/// this hand-written parser has no machinery for):
///
/// ```text
/// somelib.url = "https://github.com/user/somelib"
/// somelib.requested = "branch:main"
/// somelib.resolved = "a1b2c3d4e5f6..."
/// ```
///
/// Auto-created on first use (a missing file parses as empty, not an error — same "just
/// appears" UX as `Cargo.lock`) and only rewritten when something in it actually changed
/// (`dirty`), so an unchanged lock never gets its mtime bumped by an incidental build.
pub(crate) struct BoringLock {
    entries: HashMap<String, LockEntry>,
    dirty: bool,
}

impl BoringLock {
    pub(crate) fn load(boring_toml_dir: &Path) -> Self {
        let path = boring_toml_dir.join("boring.lock");
        let src = std::fs::read_to_string(&path).unwrap_or_default();
        Self { entries: Self::parse(&src), dirty: false }
    }

    fn parse(src: &str) -> HashMap<String, LockEntry> {
        // (url, requested, resolved) — collected per entry name before any of the three fields
        // is known to be present, so each starts `None` until its line is seen.
        type PartialFields = HashMap<String, (Option<String>, Option<String>, Option<String>)>;
        let mut fields: PartialFields = HashMap::new();
        for line in src.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            let Some((key, value)) = line.split_once('=') else { continue };
            let Some((name, field)) = key.trim().split_once('.') else { continue };
            let value = value.trim().trim_matches('"').to_string();
            let entry = fields.entry(name.to_string()).or_default();
            match field {
                "url" => entry.0 = Some(value),
                "requested" => entry.1 = Some(value),
                "resolved" => entry.2 = Some(value),
                _ => {}
            }
        }
        fields.into_iter()
            .filter_map(|(name, (url, requested, resolved))| {
                Some((name, LockEntry { url: url?, requested: requested?, resolved: resolved? }))
            })
            .collect()
    }

    /// Returns the locked commit for `name` if it's still valid for the *current* `url`/
    /// `requested` — a config change (different URL, switched branch, etc.) is detected here
    /// and treated as "no lock entry", not silently honored against stale data.
    fn get(&self, name: &str, url: &str, requested: &str) -> Option<&str> {
        self.entries.get(name)
            .filter(|e| e.url == url && e.requested == requested)
            .map(|e| e.resolved.as_str())
    }

    fn set(&mut self, name: &str, url: &str, requested: &str, resolved: &str) {
        let new_entry = LockEntry { url: url.to_string(), requested: requested.to_string(), resolved: resolved.to_string() };
        if self.entries.get(name) != Some(&new_entry) {
            self.entries.insert(name.to_string(), new_entry);
            self.dirty = true;
        }
    }

    pub(crate) fn save(&self, boring_toml_dir: &Path) -> Result<(), String> {
        if !self.dirty { return Ok(()); }
        let mut names: Vec<&String> = self.entries.keys().collect();
        names.sort(); // stable order so an unchanged lock diffs as unchanged in version control
        let mut out = String::from(
            "# boring.lock — auto-generated by `boring build`/`boring run`. Do not edit by\n\
             # hand; run `boring update [name]` to deliberately refresh an entry, or just\n\
             # delete this file to force every git dependency to re-resolve from scratch.\n"
        );
        for name in names {
            let e = &self.entries[name];
            out.push_str(&format!("{name}.url = \"{}\"\n", e.url));
            out.push_str(&format!("{name}.requested = \"{}\"\n", e.requested));
            out.push_str(&format!("{name}.resolved = \"{}\"\n", e.resolved));
        }
        let path = boring_toml_dir.join("boring.lock");
        std::fs::write(&path, out).map_err(|e| format!("cannot write '{}': {e}", path.display()))
    }
}

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
pub(crate) fn resolve_git_dep(url: &str, gitref: &GitRef, policy: DepPolicy) -> Result<PathBuf, String> {
    let dir = dep_cache_dir(url, gitref);
    if dir.join(".git").is_dir() {
        refresh(&dir, gitref, policy)?;
    } else {
        if policy.offline {
            return Err(format!(
                "git dependency '{url}' has no local cache yet and --offline was given \
                 — a first-time clone needs network access"
            ));
        }
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

/// `resolve_git_dep`, but pins a *mutable* ref (`branch`/`tag`/the default branch) to whatever
/// commit `lock` already has on file for this exact `(name, url, requested)` — avoiding both the
/// network round-trip and the drift a plain `resolve_git_dep(url, gitref)` call would allow.
/// `GitRef::Rev` bypasses the lock entirely (already exact — see `BoringLock`'s doc comment).
/// On a fresh resolution (no matching lock entry — new dependency, or `boring.toml` changed
/// what it asks for), resolves normally via the original `gitref` and records the result's
/// `HEAD` commit into `lock` for next time. Called from `BoringToml::resolve_deps`; `boring
/// update` calls the lock-unaware `resolve_git_dep` directly instead, since forcing a fresh
/// resolution regardless of any existing entry is the entire point of that command.
pub(crate) fn resolve_git_dep_with_lock(
    name: &str,
    url: &str,
    gitref: &GitRef,
    lock: &mut BoringLock,
    policy: DepPolicy,
) -> Result<PathBuf, String> {
    let GitRef::Rev(_) = gitref else {
        let requested = gitref.to_lock_string();
        if let Some(locked_sha) = lock.get(name, url, &requested) {
            return resolve_git_dep(url, &GitRef::Rev(locked_sha.to_string()), policy);
        }
        // No matching lock entry: a new dependency, or `boring.toml` now asks for something
        // different than last time. Resolving this normally would create/change a lock entry
        // — exactly what `--locked` forbids. Checked before any network attempt at all.
        if policy.locked {
            return Err(format!(
                "'{name}' has no matching boring.lock entry (new dependency, or boring.toml \
                 changed what it asks for) and --locked was given — run `boring update {name}` \
                 first, or drop --locked"
            ));
        }
        let src = resolve_git_dep(url, gitref, policy)?;
        let resolved_sha = run_git(Some(&dep_cache_dir(url, gitref)), &["rev-parse", "HEAD"], "rev-parse")?;
        lock.set(name, url, &requested, &resolved_sha);
        return Ok(src);
    };
    resolve_git_dep(url, gitref, policy)
}

/// Result of `boring update`'s per-dependency force refresh — `main.rs`'s `parse_update_command`
/// reports each of these to the user with its own message.
pub(crate) enum UpdateOutcome {
    /// A `rev`-pinned dependency — already exact, nothing for `update` to do.
    AlreadyExact,
    /// Re-resolved, but landed on the same commit already recorded in the lock.
    UpToDate(String),
    /// Re-resolved to a different commit than before (`None` if there was no prior lock entry
    /// for this exact `(name, url, requested)` — a brand new dependency, or `boring.toml`
    /// changed what it asks for).
    Updated(Option<String>, String),
}

/// Forces a fresh resolution of one git dependency, ignoring whatever the lock currently says —
/// the entire point of `boring update` (`resolve_git_dep_with_lock`, used by ordinary `boring
/// run`/`boring build`, deliberately does the opposite: prefer the lock over the network).
/// Updates `lock` in place when the resolved commit actually changes.
pub(crate) fn update_git_dep(name: &str, url: &str, gitref: &GitRef, lock: &mut BoringLock) -> Result<UpdateOutcome, String> {
    // `boring update`'s whole purpose is a fresh, unrestricted resolution — always the default
    // policy here regardless of anything an (unrelated) ambient `--locked`/`--offline` might say.
    let policy = DepPolicy::default();
    if let GitRef::Rev(_) = gitref {
        resolve_git_dep(url, gitref, policy)?; // still confirm it's actually resolvable
        return Ok(UpdateOutcome::AlreadyExact);
    }
    let requested = gitref.to_lock_string();
    let old = lock.get(name, url, &requested).map(str::to_string);
    resolve_git_dep(url, gitref, policy)?;
    let new_sha = run_git(Some(&dep_cache_dir(url, gitref)), &["rev-parse", "HEAD"], "rev-parse")?;
    if old.as_deref() == Some(new_sha.as_str()) {
        return Ok(UpdateOutcome::UpToDate(new_sha));
    }
    lock.set(name, url, &requested, &new_sha);
    Ok(UpdateOutcome::Updated(old, new_sha))
}

// Note: no `policy` parameter here — `resolve_git_dep` already refuses to reach this function
// at all when `policy.offline` is set and no cache exists yet (a fresh clone is unconditionally
// a network operation with no local fallback), so by the time this runs, offline mode either
// doesn't apply or has already been rejected.
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
fn refresh(dir: &Path, gitref: &GitRef, policy: DepPolicy) -> Result<(), String> {
    match gitref {
        GitRef::Rev(rev) => {
            if commit_present(dir, rev) {
                return run_git(Some(dir), &["checkout", rev], "checkout").map(|_| ());
            }
            if policy.offline {
                return Err(format!(
                    "commit '{rev}' is not present in the local cache and --offline was given \
                     — fetching it needs network access"
                ));
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
        GitRef::Branch(b) => refresh_mutable_ref(dir, Some(b), policy),
        GitRef::Tag(t) => refresh_mutable_ref(dir, Some(t), policy),
        GitRef::Default => refresh_mutable_ref(dir, None, policy),
    }
}

fn refresh_mutable_ref(dir: &Path, ref_name: Option<&str>, policy: DepPolicy) -> Result<(), String> {
    if policy.offline {
        // Skip the fetch attempt entirely rather than trying-then-falling-back (today's
        // non-offline behavior below) -- `--offline` means "don't even try the network", not
        // "try quietly and swallow the failure". Whatever's already checked out from a prior
        // resolution is used as-is; nothing to check out again for a mutable ref (there's no
        // "pending" checkout to apply -- the last successful fetch already left HEAD there).
        return if run_git(Some(dir), &["rev-parse", "--verify", "-q", "HEAD"], "rev-parse").is_ok() {
            Ok(())
        } else {
            Err("local cache exists but has no usable commit checked out, and --offline was given".to_string())
        };
    }
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

    /// Reads a file and normalizes line endings to `\n`. On Windows, `git clone`/`checkout`
    /// into the destination cache dir applies the *destination* repo's `core.autocrlf` (which
    /// defaults from the machine's global git config, often `true`) regardless of how the
    /// fixture repo itself was configured -- so a file written with literal `\n` can come back
    /// as `\r\n` after a real clone/checkout round-trip. Tests that assert against a literal
    /// `\n`-only string must read through this instead of `std::fs::read_to_string` directly.
    fn read_normalized(path: &Path) -> String {
        std::fs::read_to_string(path).unwrap().replace("\r\n", "\n")
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
        let src1 = resolve_git_dep(&repo_url, &GitRef::Default, DepPolicy::default()).expect("default-branch clone");
        assert!(src1.join("lib.br").is_file());

        // Resolving again must succeed without the fixture doing anything further -- proves
        // the cache directory is actually being read on the second call.
        let src1_again = resolve_git_dep(&repo_url, &GitRef::Default, DepPolicy::default()).expect("cached re-resolve");
        assert_eq!(src1, src1_again);

        // A specific rev, in a *separate* cache directory from the default-branch one (per
        // dep_cache_dir's one-directory-per-(url,ref) design).
        let src2 = resolve_git_dep(&repo_url, &GitRef::Rev(sha.clone()), DepPolicy::default()).expect("rev-pinned clone");
        assert!(src2.join("lib.br").is_file());
        assert_ne!(src1, src2);

        // Re-resolving the same rev must need no network at all (commit_present short-circuits
        // straight to checkout) -- if this were broken it would try to fetch from `repo_url`
        // and still succeed here since the fixture still exists, so this mainly guards against
        // a panic/error in that code path, not staleness; the important behavioral guarantee
        // (no fetch attempted) is documented in `refresh`'s doc comment.
        let src2_again = resolve_git_dep(&repo_url, &GitRef::Rev(sha), DepPolicy::default()).expect("cached rev re-resolve");
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
        let err = resolve_git_dep(&repo.to_string_lossy(), &GitRef::Default, DepPolicy::default()).unwrap_err();
        assert!(err.contains("no 'src/' directory"), "unexpected error: {err}");
        std::env::remove_var("BORING_CACHE_DIR");

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn offline_policy_blocks_fresh_clone_but_allows_cached_reuse() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = std::env::temp_dir().join(format!("boring_offline_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let repo = tmp.join("repo");
        make_fixture_repo(&repo);
        let repo_url = repo.to_string_lossy().to_string();
        let cache = tmp.join("cache");
        std::env::set_var("BORING_CACHE_DIR", &cache);

        let offline = DepPolicy { locked: false, offline: true };

        // Nothing cached yet -- a fresh clone needs network, --offline must refuse it.
        let err = resolve_git_dep(&repo_url, &GitRef::Default, offline).unwrap_err();
        assert!(err.contains("--offline"), "unexpected error: {err}");

        // Clone it for real (no policy restriction)...
        let src = resolve_git_dep(&repo_url, &GitRef::Default, DepPolicy::default()).unwrap();
        assert!(src.join("lib.br").is_file());
        // ...then break the "remote" -- proves a subsequent --offline resolution genuinely
        // never touches it again (if it tried to fetch, this would fail instead of succeeding).
        std::fs::rename(&repo, tmp.join("repo_gone")).unwrap();

        let src_offline = resolve_git_dep(&repo_url, &GitRef::Default, offline)
            .expect("an already-cached dependency must resolve offline with no fetch attempt");
        assert_eq!(src, src_offline);

        std::env::remove_var("BORING_CACHE_DIR");
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn locked_policy_blocks_new_lock_entries_but_allows_matching_ones() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = std::env::temp_dir().join(format!("boring_locked_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let repo = tmp.join("repo");
        make_fixture_repo(&repo);
        let repo_url = repo.to_string_lossy().to_string();
        let project = tmp.join("project");
        std::fs::create_dir_all(&project).unwrap();
        let cache = tmp.join("cache");
        std::env::set_var("BORING_CACHE_DIR", &cache);

        let gitref = GitRef::Branch("main".to_string());
        let locked = DepPolicy { locked: true, offline: false };

        // No lock entry yet -- --locked must refuse to create one, and must not touch the lock.
        let mut lock = BoringLock::load(&project);
        let err = resolve_git_dep_with_lock("numlib", &repo_url, &gitref, &mut lock, locked).unwrap_err();
        assert!(err.contains("--locked"), "unexpected error: {err}");
        assert!(!lock.dirty, "a refused resolution must not have touched the lock");

        // Resolve normally once to create a matching entry, and persist it...
        let mut lock2 = BoringLock::load(&project);
        let src = resolve_git_dep_with_lock("numlib", &repo_url, &gitref, &mut lock2, DepPolicy::default()).unwrap();
        lock2.save(&project).unwrap();

        // ...then --locked must succeed against that existing, matching entry. Compares
        // *content*, not the path itself: the locked path resolves through a synthesized
        // `GitRef::Rev(locked_sha)`, which lives in its own cache directory (keyed by
        // `(url, gitref)`, per `dep_cache_dir`'s design) distinct from the `Branch("main")`
        // directory `src` came from, even though both hold the same commit's content.
        let mut lock3 = BoringLock::load(&project);
        let src_locked = resolve_git_dep_with_lock("numlib", &repo_url, &gitref, &mut lock3, locked)
            .expect("a matching lock entry must still resolve under --locked");
        assert_eq!(
            std::fs::read_to_string(src.join("lib.br")).unwrap(),
            std::fs::read_to_string(src_locked.join("lib.br")).unwrap()
        );

        std::env::remove_var("BORING_CACHE_DIR");
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn boring_lock_parse_and_save_round_trip() {
        let tmp = std::env::temp_dir().join(format!("boring_lock_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let mut lock = BoringLock::load(&tmp); // no file yet -- parses as empty, not an error
        assert!(lock.entries.is_empty());
        assert!(!lock.dirty);

        lock.set("somelib", "https://example.com/x", "branch:main", "abc123");
        assert!(lock.dirty);
        lock.save(&tmp).unwrap();

        let reloaded = BoringLock::load(&tmp);
        assert_eq!(
            reloaded.get("somelib", "https://example.com/x", "branch:main"),
            Some("abc123")
        );
        // A config change (different requested ref) must not reuse the stale entry.
        assert_eq!(reloaded.get("somelib", "https://example.com/x", "branch:other"), None);
        // An unchanged reload has nothing new to save.
        assert!(!reloaded.dirty);

        std::fs::remove_dir_all(&tmp).ok();
    }

    // The actual reproducibility guarantee `boring.lock` exists for: a `branch`-tracking
    // dependency stays pinned to whatever commit it last resolved to, even after the upstream
    // repo moves on, until `boring update` is run deliberately. Real local git repo (no
    // network) advanced between two resolutions to prove this isn't just cache reuse.
    #[test]
    fn branch_dependency_stays_pinned_until_update() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let tmp = std::env::temp_dir().join(format!("boring_lock_pin_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let repo = tmp.join("repo");
        let first_sha = make_fixture_repo(&repo);
        let repo_url = repo.to_string_lossy().to_string();

        let project = tmp.join("project");
        std::fs::create_dir_all(&project).unwrap();
        let cache = tmp.join("cache");
        std::env::set_var("BORING_CACHE_DIR", &cache);

        let gitref = GitRef::Branch("main".to_string());
        let mut lock = BoringLock::load(&project);
        let src1 = resolve_git_dep_with_lock("numlib", &repo_url, &gitref, &mut lock, DepPolicy::default()).unwrap();
        assert_eq!(
            read_normalized(&src1.join("lib.br")),
            "pub struct Marker:\n    pub int value\n"
        );
        lock.save(&project).unwrap();
        assert_eq!(
            BoringLock::load(&project).get("numlib", &repo_url, "branch:main"),
            Some(first_sha.as_str())
        );

        // The upstream repo moves on...
        std::fs::write(repo.join("src").join("lib.br"), "pub struct Marker:\n    pub int value2\n").unwrap();
        run(&repo, &["commit", "-aq", "-m", "second"]);

        // ...but a fresh `resolve_git_dep_with_lock` call (same as an ordinary `boring run`/
        // `boring build`) must still return the ORIGINAL content -- proves genuine pinning,
        // not just "the cache directory happens to already exist".
        let mut lock2 = BoringLock::load(&project);
        let src2 = resolve_git_dep_with_lock("numlib", &repo_url, &gitref, &mut lock2, DepPolicy::default()).unwrap();
        assert_eq!(
            read_normalized(&src2.join("lib.br")),
            "pub struct Marker:\n    pub int value\n",
            "a locked branch dependency must not silently drift to the new commit"
        );
        assert!(!lock2.dirty, "nothing changed, so re-resolving must not rewrite the lock");

        // `boring update` forces past the lock and picks up the new commit.
        match update_git_dep("numlib", &repo_url, &gitref, &mut lock2).unwrap() {
            UpdateOutcome::Updated(Some(old), new) => {
                assert_eq!(old, first_sha);
                assert_ne!(new, first_sha);
            }
            _other => panic!("expected Updated(Some(_), _), got a different outcome"),
        }
        lock2.save(&project).unwrap();

        let src3 = resolve_git_dep_with_lock("numlib", &repo_url, &gitref, &mut BoringLock::load(&project), DepPolicy::default()).unwrap();
        assert_eq!(
            read_normalized(&src3.join("lib.br")),
            "pub struct Marker:\n    pub int value2\n",
            "after `boring update`, the lock must reflect the new commit"
        );

        std::env::remove_var("BORING_CACHE_DIR");
        std::fs::remove_dir_all(&tmp).ok();
    }
}
