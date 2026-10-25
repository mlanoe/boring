// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Run-time backend selection policy for multi-target builds (`boring build --target all`).
//
// This file is standard-library only, and is used twice: it is copied verbatim into every
// generated multi-target crate as `src/policy.rs`, and it is compiled into `boring` itself under
// `cfg(test)` so the ranking, cache and load logic below are unit-tested.
//
// Policy: every usable backend is benchmarked once (a fixed compute-bound kernel, result in
// millions of multiply-adds per second), the results are cached per device, and the backends are
// ranked by `score / (1 + load)`, where `load` is how many other boring processes currently
// hold a lease on that backend. A backend that cannot be benchmarked gets a fixed prior.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const CACHE_VERSION: &str = "v1";
const CACHE_FILE: &str = "backends.tsv";
/// A cached score older than this is measured again.
const MAX_AGE_SECS: u64 = 30 * 24 * 3600;
/// Highest load a backend can report (one lease slot per concurrent process).
const LEASE_SLOTS: usize = 8;

/// Assumed score for a backend that was not (or could not be) measured. Native vendor backends
/// are assumed faster than the portable and CPU ones; a measured score replaces the prior.
pub fn prior(backend: &str) -> f64 {
    match backend {
        "cuda" => 10_000_000.0,
        "rocm" => 8_000_000.0,
        "metal" => 2_000_000.0,
        "wgpu" => 1_000_000.0,
        _ => 100_000.0,
    }
}

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// `BORING_CACHE_DIR`, else the platform's per-user cache directory.
pub fn cache_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("BORING_CACHE_DIR") {
        return Some(PathBuf::from(dir));
    }
    if cfg!(windows) {
        return std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("boring"));
    }
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        return Some(PathBuf::from(dir).join("boring"));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache").join("boring"))
}

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub device: String,
    /// `None` records a backend that could not be benchmarked (it is not retried until it expires).
    pub score: Option<f64>,
    pub stamp: u64,
}

impl Entry {
    /// Still valid for `device` at time `at`.
    pub fn is_fresh(&self, device: &str, at: u64) -> bool {
        self.device == clean(device) && at.saturating_sub(self.stamp) < MAX_AGE_SECS
    }
}

fn clean(text: &str) -> String {
    text.replace(['\t', '\n', '\r'], " ")
}

pub fn load_cache(dir: &Path) -> BTreeMap<String, Entry> {
    let mut cache = BTreeMap::new();
    let Ok(text) = fs::read_to_string(dir.join(CACHE_FILE)) else { return cache };
    for line in text.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        if let [version, backend, device, score, stamp] = fields[..] {
            if version != CACHE_VERSION {
                continue;
            }
            let (Ok(stamp), score) = (stamp.parse::<u64>(), score.parse::<f64>().ok()) else { continue };
            cache.insert(backend.to_string(), Entry { device: device.to_string(), score, stamp });
        }
    }
    cache
}

/// Best effort: a read-only or missing cache directory only means benchmarks are repeated.
pub fn save_cache(dir: &Path, cache: &BTreeMap<String, Entry>) {
    if fs::create_dir_all(dir).is_err() {
        return;
    }
    let mut text = String::new();
    for (backend, entry) in cache {
        let score = entry.score.map(|s| s.to_string()).unwrap_or_else(|| "-".into());
        text.push_str(&format!("{CACHE_VERSION}\t{backend}\t{}\t{score}\t{}\n", clean(&entry.device), entry.stamp));
    }
    let tmp = dir.join(format!("{CACHE_FILE}.{}.tmp", std::process::id()));
    if fs::write(&tmp, text).is_ok() && fs::rename(&tmp, dir.join(CACHE_FILE)).is_err() {
        let _ = fs::remove_file(&tmp);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ranked {
    pub name: String,
    pub score: f64,
    pub measured: bool,
    pub load: usize,
    pub effective: f64,
}

/// A backend earlier in the static preference order wins over a later one that is faster by
/// less than this margin: measurements are noisy, and a native backend is the safer pick
/// (Metal over wgpu-on-Metal, for instance).
const TIE_MARGIN: f64 = 0.8;

/// Rank `(backend, measured score, load)` candidates, best first. Among the candidates within
/// `TIE_MARGIN` of the best remaining one, the one earliest in the input order goes first.
pub fn rank(candidates: &[(String, Option<f64>, usize)]) -> Vec<Ranked> {
    let mut remaining: Vec<Ranked> = candidates
        .iter()
        .map(|(name, score, load)| {
            let measured = score.is_some();
            let score = score.unwrap_or_else(|| prior(name));
            Ranked { name: name.clone(), score, measured, load: *load, effective: score / (1.0 + *load as f64) }
        })
        .collect();
    let mut ranked = Vec::with_capacity(remaining.len());
    while !remaining.is_empty() {
        let best = remaining.iter().map(|r| r.effective).fold(f64::MIN, f64::max);
        let pick = remaining.iter().position(|r| r.effective >= best * TIE_MARGIN).unwrap_or(0);
        ranked.push(remaining.remove(pick));
    }
    ranked
}

fn slot_path(dir: &Path, backend: &str, slot: usize) -> PathBuf {
    dir.join("leases").join(format!("{backend}.{slot}.lock"))
}

fn open_slot(dir: &Path, backend: &str, slot: usize) -> Option<File> {
    let path = slot_path(dir, backend, slot);
    fs::create_dir_all(path.parent()?).ok()?;
    OpenOptions::new().create(true).truncate(false).write(true).open(path).ok()
}

/// How many other processes currently hold a lease on `backend`. A lease is an OS file lock, so
/// it disappears by itself when its process exits or crashes.
pub fn current_load(dir: &Path, backend: &str) -> usize {
    (0..LEASE_SLOTS)
        .filter(|&slot| match open_slot(dir, backend, slot) {
            Some(file) => matches!(file.try_lock(), Err(TryLockError::WouldBlock)),
            None => false,
        })
        .count()
}

/// A held lease on a backend; dropping it (or exiting) releases it.
pub struct Lease {
    _file: Option<File>,
}

/// Claim the first free lease slot of `backend`. When every slot is taken (or the cache directory
/// is unusable) the returned lease is empty: the load is simply not recorded.
pub fn acquire(dir: &Path, backend: &str) -> Lease {
    for slot in 0..LEASE_SLOTS {
        if let Some(file) = open_slot(dir, backend, slot) {
            if file.try_lock().is_ok() {
                return Lease { _file: Some(file) };
            }
        }
    }
    Lease { _file: None }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("boring_policy_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn cand(name: &str, score: Option<f64>, load: usize) -> (String, Option<f64>, usize) {
        (name.to_string(), score, load)
    }

    #[test]
    fn faster_backend_wins_and_load_demotes_it() {
        let order = |load_metal| {
            rank(&[cand("metal", Some(25_000.0), load_metal), cand("wgpu", Some(20_000.0), 0), cand("simd", Some(2_000.0), 0)])
                .into_iter()
                .map(|r| r.name)
                .collect::<Vec<_>>()
        };
        assert_eq!(order(0), ["metal", "wgpu", "simd"]);
        // Within the tie margin the earlier (native) backend stays first.
        let close = rank(&[cand("metal", Some(237.0), 0), cand("wgpu", Some(287.0), 0)]);
        assert_eq!(close[0].name, "metal");
        assert_eq!(order(1), ["wgpu", "metal", "simd"]);
        assert_eq!(order(3), ["wgpu", "metal", "simd"]);
        assert_eq!(order(20), ["wgpu", "simd", "metal"]);
    }

    #[test]
    fn unmeasured_backend_uses_its_prior_and_ties_keep_order() {
        let ranked = rank(&[cand("simd", Some(2_000.0), 0), cand("cuda", None, 0)]);
        assert_eq!(ranked[0].name, "cuda");
        assert!(!ranked[0].measured && ranked[0].score == prior("cuda"));
        let tied = rank(&[cand("a", Some(5.0), 0), cand("b", Some(5.0), 0)]);
        assert_eq!(tied[0].name, "a");
    }

    #[test]
    fn cache_round_trips_and_expires() {
        let dir = scratch("cache");
        let mut cache = BTreeMap::new();
        cache.insert("metal".to_string(), Entry { device: "Apple\tM3".into(), score: Some(123.5), stamp: 1_000 });
        cache.insert("cuda".to_string(), Entry { device: "RTX".into(), score: None, stamp: 1_000 });
        save_cache(&dir, &cache);
        let loaded = load_cache(&dir);
        assert_eq!(loaded["metal"].score, Some(123.5));
        assert_eq!(loaded["cuda"].score, None);
        assert!(loaded["metal"].is_fresh("Apple\tM3", 2_000));
        assert!(!loaded["metal"].is_fresh("Apple M4", 2_000), "another device invalidates");
        assert!(!loaded["metal"].is_fresh("Apple M3", 1_000 + MAX_AGE_SECS), "old entries expire");
        let _ = fs::remove_dir_all(&dir);
    }

    fn settles_at(dir: &Path, backend: &str, expected: usize) -> bool {
        for _ in 0..200 {
            if current_load(dir, backend) == expected {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        false
    }

    #[test]
    fn leases_count_concurrent_holders() {
        let dir = scratch("lease");
        assert!(settles_at(&dir, "wgpu", 0), "load of wgpu stayed at {}", current_load(&dir, "wgpu"));
        let first = acquire(&dir, "wgpu");
        assert!(settles_at(&dir, "wgpu", 1), "load of wgpu stayed at {}", current_load(&dir, "wgpu"));
        let second = acquire(&dir, "wgpu");
        assert!(settles_at(&dir, "wgpu", 2), "load of wgpu stayed at {}", current_load(&dir, "wgpu"));
        assert!(settles_at(&dir, "simd", 0), "leases are per backend");
        // A lock is released when the last copy of its file descriptor closes. A child process
        // another test forks at this very moment still holds a copy until its `exec`, so the
        // release can be observed a few microseconds late: wait for it instead of racing it.
        drop(first);
        assert!(settles_at(&dir, "wgpu", 1), "load stayed at {}", current_load(&dir, "wgpu"));
        drop(second);
        assert!(settles_at(&dir, "wgpu", 0), "load stayed at {}", current_load(&dir, "wgpu"));
        let _ = fs::remove_dir_all(&dir);
    }
}
