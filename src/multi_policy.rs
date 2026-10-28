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

/// Assumed score for a backend whose device is not in [`DEVICES`] and that was not (or could not
/// be) measured. Native vendor backends are assumed faster than the portable and CPU ones; a
/// measured score replaces the prior. Same scale as the measured scores (see `SCORE_PER_TFLOPS`).
pub fn prior(backend: &str) -> f64 {
    match backend {
        "cuda" => 1_200_000.0,
        "rocm" => 900_000.0,
        "metal" => 400_000.0,
        "wgpu" => 250_000.0,
        _ => 100_000.0,
    }
}

/// How many benchmark score points one TFLOPS of peak FP32 throughput is worth. The benchmark is
/// a chain of dependent multiply-adds, so it reaches a fraction of the peak: measured on an
/// Apple M3 (4.1 TFLOPS) it scores 260 thousand on Metal, about 63 thousand per TFLOPS.
const SCORE_PER_TFLOPS: f64 = 60_000.0;

/// wgpu reaches about 60% of what the same device scores through its native API (an M3 scores
/// 156 thousand on wgpu against 260 thousand on Metal), so a native backend wins a tie.
const WGPU_FACTOR: f64 = 0.6;

/// Approximate peak FP32 TFLOPS of known devices, from the vendors' public figures: the first
/// entry whose pattern the lower-cased device name contains wins, so a longer name comes before
/// the shorter one it contains (`rtx 4070 ti` before `rtx 4070`). It ranks backends against each
/// other when nothing was measured (`BORING_BENCH=off`, a backend that cannot be benchmarked such
/// as ROCm). Memory size, clocks and core counts change the real figure by tens of percent for
/// some entries: the order of very different devices is meaningful, not that of close ones.
const DEVICES: &[(&str, f64)] = &[
    // NVIDIA GeForce RTX 50 / 40 / 30 / 20, GTX 16 / 10
    ("rtx 5090", 104.8), ("rtx 5080", 56.3), ("rtx 5070 ti", 43.9), ("rtx 5070", 30.9), ("rtx 5060 ti", 23.7), ("rtx 5060", 19.2),
    ("rtx 4090", 82.6), ("rtx 4080", 48.7), ("rtx 4070 ti", 40.1), ("rtx 4070", 29.1), ("rtx 4060 ti", 22.1), ("rtx 4060", 15.1),
    ("rtx 3090 ti", 40.0), ("rtx 3090", 35.6), ("rtx 3080 ti", 34.1), ("rtx 3080", 29.8), ("rtx 3070 ti", 21.7), ("rtx 3070", 20.3),
    ("rtx 3060 ti", 16.2), ("rtx 3060", 12.7), ("rtx 3050", 9.1),
    ("rtx 2080 ti", 13.4), ("rtx 2080", 10.1), ("rtx 2070", 7.5), ("rtx 2060", 6.5),
    ("gtx 1660", 5.0), ("gtx 1080 ti", 11.3), ("gtx 1080", 8.9), ("gtx 1070", 6.5), ("gtx 1060", 4.4), ("gtx 1050", 1.9),
    // NVIDIA data centre and workstation
    ("h200", 67.0), ("h100", 60.0), (" l40", 90.5), (" l4", 30.3), (" a100", 19.5), (" a40", 37.4), (" a10", 31.2), (" t4", 8.1),
    (" v100", 15.7), (" p100", 9.3), ("rtx 6000 ada", 91.1), ("rtx a6000", 38.7),
    // AMD Radeon RX 7000 / 6000 / older, Instinct
    ("rx 7900 xtx", 61.4), ("rx 7900 xt", 51.5), ("rx 7900 gre", 46.0), ("rx 7800 xt", 37.3), ("rx 7700 xt", 35.2), ("rx 7600", 21.8),
    ("rx 6950 xt", 23.7), ("rx 6900 xt", 23.0), ("rx 6800 xt", 20.7), ("rx 6800", 16.2), ("rx 6750 xt", 13.3), ("rx 6700 xt", 13.2),
    ("rx 6650 xt", 10.8), ("rx 6600 xt", 10.6), ("rx 6600", 8.9), ("rx 6500", 5.8), ("rx 580", 6.2), ("rx 570", 5.1),
    ("radeon vii", 13.8), ("vega 64", 12.7), ("mi300", 163.0), ("mi250", 45.3), ("mi210", 22.6), ("mi100", 23.1),
    // Intel Arc
    ("arc a770", 19.7), ("arc a750", 17.2), ("arc a580", 12.3), ("arc a380", 4.2), ("arc b580", 14.6), ("arc b570", 12.3),
    // Apple silicon (the larger configurations of each family)
    ("apple m4 max", 18.4), ("apple m4 pro", 9.2), ("apple m4", 4.3),
    ("apple m3 ultra", 28.2), ("apple m3 max", 16.4), ("apple m3 pro", 7.4), ("apple m3", 4.1),
    ("apple m2 ultra", 27.2), ("apple m2 max", 13.6), ("apple m2 pro", 6.8), ("apple m2", 3.6),
    ("apple m1 ultra", 21.0), ("apple m1 max", 10.4), ("apple m1 pro", 5.2), ("apple m1", 2.6),
];

/// The assumed score of `backend` running on `device`: the embedded table when it knows the
/// device, else the backend's [`prior`].
pub fn device_prior(backend: &str, device: &str) -> f64 {
    if backend == "simd" {
        return prior(backend);
    }
    let name = device.to_lowercase();
    match DEVICES.iter().find(|(pattern, _)| name.contains(pattern)) {
        Some((_, tflops)) => tflops * SCORE_PER_TFLOPS * if backend == "wgpu" { WGPU_FACTOR } else { 1.0 },
        None => prior(backend),
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
/// `candidates` are `(backend, measured score, load, assumed score when not measured)`.
pub fn rank(candidates: &[(String, Option<f64>, usize, f64)]) -> Vec<Ranked> {
    let mut remaining: Vec<Ranked> = candidates
        .iter()
        .map(|(name, score, load, assumed)| {
            let measured = score.is_some();
            let score = score.unwrap_or(*assumed);
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

    fn cand(name: &str, score: Option<f64>, load: usize) -> (String, Option<f64>, usize, f64) {
        (name.to_string(), score, load, prior(name))
    }

    #[test]
    fn known_devices_are_ranked_by_the_embedded_table() {
        let score = |backend, device| device_prior(backend, device);
        // A longer name beats the shorter one it contains.
        assert!(score("cuda", "NVIDIA GeForce RTX 4070 Ti") > score("cuda", "NVIDIA GeForce RTX 4070"));
        assert!(score("cuda", "NVIDIA GeForce RTX 4090") > score("rocm", "AMD Radeon RX 7900 XTX"));
        assert!(score("rocm", "AMD Radeon RX 6600") > score("metal", "Apple M3"));
        assert!(score("metal", "Apple M3 Max") > score("metal", "Apple M3 Pro") && score("metal", "Apple M3 Pro") > score("metal", "Apple M3"));
        // The short data-centre names do not match inside longer ones.
        assert_eq!(score("cuda", "NVIDIA A100-SXM4-40GB"), 19.5 * SCORE_PER_TFLOPS);
        assert_eq!(score("cuda", "NVIDIA L40S"), 90.5 * SCORE_PER_TFLOPS);
        assert_eq!(score("cuda", "NVIDIA L4"), 30.3 * SCORE_PER_TFLOPS);
        // The same device through wgpu is worth less than through its native API.
        assert!(score("wgpu", "Apple M3") < score("metal", "Apple M3"));
        // Unknown devices and the CPU fall back to the backend's prior.
        assert_eq!(score("cuda", "Mystery GPU 9000"), prior("cuda"));
        assert_eq!(score("simd", "RTX 4090"), prior("simd"));
        // Every entry is reachable: no pattern is hidden behind an earlier, shorter one.
        for (i, (pattern, _)) in DEVICES.iter().enumerate() {
            let first = DEVICES.iter().position(|(p, _)| pattern.contains(p)).unwrap();
            assert_eq!(first, i, "`{pattern}` is shadowed by `{}`", DEVICES[first].0);
        }
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
