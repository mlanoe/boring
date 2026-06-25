// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// GPU simulation profile — loaded from a TOML file or one of the built-in profiles.
//
// Built-in profiles (select with `boring run --gpu <name> file.br`):
//   default  — generic simulated GPU
//   v100     — Tesla V100 SXM2 16GB
//   a100     — A100 SXM4 80GB
//   rtx3090  — GeForce RTX 3090 24GB
//   rtx4090  — GeForce RTX 4090 24GB
//   h100     — H100 SXM5 80GB
//
// Custom profile: pass a path to a TOML file.

// ─── Embedded profile sources ─────────────────────────────────────────────────

const PROFILE_DEFAULT: &str = include_str!("../../gpu-profiles/default.toml");
const PROFILE_V100:    &str = include_str!("../../gpu-profiles/v100.toml");
const PROFILE_A100:    &str = include_str!("../../gpu-profiles/a100.toml");
const PROFILE_RTX3090: &str = include_str!("../../gpu-profiles/rtx3090.toml");
const PROFILE_RTX4090: &str = include_str!("../../gpu-profiles/rtx4090.toml");
const PROFILE_H100:    &str = include_str!("../../gpu-profiles/h100.toml");

// ─── Profile struct ───────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct GpuProfile {
    pub name:                String,
    pub total_mem:           i64,
    pub warp_size:           i64,
    pub max_threads:         i64,
    pub max_shared_mem:      i64,
    pub compute_capability:  (i64, i64),
}

impl Default for GpuProfile {
    fn default() -> Self {
        Self::parse(PROFILE_DEFAULT)
            .expect("built-in default profile is invalid")
    }
}

impl GpuProfile {
    /// Load a named built-in profile or a custom TOML file path.
    pub fn load(name: &str) -> Result<Self, String> {
        let source = match name {
            "default" => PROFILE_DEFAULT.to_string(),
            "v100"    => PROFILE_V100.to_string(),
            "a100"    => PROFILE_A100.to_string(),
            "rtx3090" => PROFILE_RTX3090.to_string(),
            "rtx4090" => PROFILE_RTX4090.to_string(),
            "h100"    => PROFILE_H100.to_string(),
            path => std::fs::read_to_string(path)
                .map_err(|e| format!("cannot read GPU profile '{}': {}", path, e))?,
        };
        Self::parse(&source)
    }

    /// Parse a TOML subset: string, integer, and [int, int] array values.
    fn parse(source: &str) -> Result<Self, String> {
        let mut name: Option<String>  = None;
        let mut total_mem: Option<i64>  = None;
        let mut warp_size: Option<i64>  = None;
        let mut max_threads: Option<i64> = None;
        let mut max_shared_mem: Option<i64> = None;
        let mut compute_capability: Option<(i64, i64)> = None;

        for raw_line in source.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }

            let (key, raw_val) = line.split_once('=')
                .ok_or_else(|| format!("invalid profile line: {raw_line}"))?;
            let key = key.trim();
            let val = raw_val.trim();

            // Strip inline comment after value
            let val = strip_comment(val);

            match key {
                "name" => {
                    name = Some(val.trim_matches('"').to_string());
                }
                "totalMem" => {
                    total_mem = Some(parse_int(val)?);
                }
                "warpSize" => {
                    warp_size = Some(parse_int(val)?);
                }
                "maxThreads" => {
                    max_threads = Some(parse_int(val)?);
                }
                "maxSharedMem" => {
                    max_shared_mem = Some(parse_int(val)?);
                }
                "computeCapability" => {
                    compute_capability = Some(parse_int_pair(val)?);
                }
                other => return Err(format!("unknown GPU profile key: '{other}'")),
            }
        }

        Ok(Self {
            name:               name.unwrap_or_else(|| "Simulated GPU".into()),
            total_mem:          total_mem.ok_or("missing 'totalMem'")?,
            warp_size:          warp_size.ok_or("missing 'warpSize'")?,
            max_threads:        max_threads.ok_or("missing 'maxThreads'")?,
            max_shared_mem:     max_shared_mem.ok_or("missing 'maxSharedMem'")?,
            compute_capability: compute_capability.ok_or("missing 'computeCapability'")?,
        })
    }
}

// ─── Parsing helpers ──────────────────────────────────────────────────────────

fn strip_comment(s: &str) -> &str {
    // Find the first '#' that is not inside a string literal
    let mut in_str = false;
    for (i, ch) in s.char_indices() {
        match ch {
            '"' => in_str = !in_str,
            '#' if !in_str => return s[..i].trim_end(),
            _ => {}
        }
    }
    s.trim_end()
}

fn parse_int(s: &str) -> Result<i64, String> {
    // Remove underscores (TOML numeric separator)
    let clean: String = s.chars().filter(|&c| c != '_').collect();
    clean.parse::<i64>().map_err(|_| format!("expected integer, got '{s}'"))
}

fn parse_int_pair(s: &str) -> Result<(i64, i64), String> {
    // Expect: [major, minor]
    let inner = s.trim().trim_start_matches('[').trim_end_matches(']');
    let parts: Vec<&str> = inner.split(',').collect();
    if parts.len() != 2 {
        return Err(format!("expected [major, minor], got '{s}'"));
    }
    Ok((parse_int(parts[0].trim())?, parse_int(parts[1].trim())?))
}
