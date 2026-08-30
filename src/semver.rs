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

//! A minimal, hand-rolled semver-*shaped* parser/matcher for `[deps]`'s optional `version`
//! requirement (see `docs/book.md` §15 and `docs/cross-project-code-sharing-gap.md`) — no crate
//! (this project has exactly two: `thiserror`/`rayon`), matching `git_deps.rs`'s own "tiny, no
//! dependency" style.
//!
//! Deliberately **not** a real dependency resolver: there is no registry, no list of candidate
//! versions to search over for a `path`/`git` dependency — a `[deps]` entry already names one
//! exact, singular target. A `version` requirement is a **compatibility assertion**, checked
//! once against whatever that single target actually declares — not a constraint fed into a
//! solver that could pick a different, better-fitting version instead. Only `^`/`~`/`=` are
//! supported (no `>=`/`<`/comma-separated ranges/pre-release tags) — real Cargo semver has all
//! of that; Boring's tiny personal-scale use case doesn't need it.

/// A parsed `major.minor.patch` version — from a `[project] version` string, which
/// `BoringToml::parse` (`src/main.rs`) always defaults to `"0.1.0"` when unset, so any
/// dependency with a `boring.toml` always has one to check against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Version {
    pub(crate) major: u64,
    pub(crate) minor: u64,
    pub(crate) patch: u64,
}

impl Version {
    /// Parses `"1.2.3"`, tolerant of a partial `"1.2"` or bare `"1"` (missing components
    /// default to 0) — real-world `[project] version` strings are often partial.
    pub(crate) fn parse(s: &str) -> Result<Version, String> {
        let s = s.trim();
        let mut parts = s.split('.');
        let parse_component = |p: Option<&str>| -> Result<u64, String> {
            match p {
                None => Ok(0),
                Some(p) => p.trim().parse::<u64>().map_err(|_| {
                    format!("'{s}' is not a valid version (expected major.minor.patch, e.g. '1.2.3')")
                }),
            }
        };
        let major = parse_component(parts.next())?;
        let minor = parse_component(parts.next())?;
        let patch = parse_component(parts.next())?;
        if parts.next().is_some() {
            return Err(format!("'{s}' is not a valid version (too many '.'-separated components)"));
        }
        Ok(Version { major, minor, patch })
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// A `[deps]` `version` requirement — see this module's own doc comment for what this does and
/// (just as importantly) does not do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum VersionReq {
    /// `=1.2.3` — must match exactly.
    Exact(Version),
    /// `^1.2.3` (also the default for a bare `"1.2.3"`, same as Cargo) — "compatible with,"
    /// where what counts as a breaking change follows Cargo's own left-most-nonzero rule: the
    /// left-most nonzero component is the one a version bump in that position is allowed to
    /// change; anything left of it is fixed, and anything considered zero when this requirement
    /// was written may not even be present in the candidate (a caret only ever *raises*
    /// trailing components, never crosses a fixed one).
    Caret(Version),
    /// `~1.2.3` — patch-level changes only if a minor version is given (`~1.2.3`/`~1.2` allow
    /// `1.2.x`); `~1` allows any `1.x.y`. The `bool` is `true` when the requirement string gave
    /// only a major component (`~1`, as opposed to `~1.2`/`~1.2.3`) — Cargo's tilde behaves
    /// differently in that case (any minor/patch, not just a fixed minor), and `Version` itself
    /// has already lost that distinction by the time it's parsed (a bare `"1"` and `"1.0"` parse
    /// to the same `Version`), so it has to be carried alongside.
    Tilde(Version, bool),
}

impl VersionReq {
    pub(crate) fn parse(s: &str) -> Result<VersionReq, String> {
        let s = s.trim();
        if let Some(rest) = s.strip_prefix('=') {
            return Ok(VersionReq::Exact(Version::parse(rest)?));
        }
        if let Some(rest) = s.strip_prefix('^') {
            return Ok(VersionReq::Caret(Version::parse(rest)?));
        }
        if let Some(rest) = s.strip_prefix('~') {
            let major_only = !rest.trim().contains('.');
            return Ok(VersionReq::Tilde(Version::parse(rest)?, major_only));
        }
        // A bare version defaults to caret, same as Cargo's own `"1.2.3"` dependency shorthand.
        Ok(VersionReq::Caret(Version::parse(s)?))
    }

    pub(crate) fn matches(&self, v: &Version) -> bool {
        match self {
            VersionReq::Exact(req) => v == req,
            VersionReq::Caret(req) => {
                if v < req { return false; }
                // Upper bound: the first nonzero component of `req` (left to right) is the one
                // a compatible version may still vary in; anything to its right is unconstrained
                // upward, anything to its left must match `req` exactly.
                if req.major > 0 {
                    v.major == req.major
                } else if req.minor > 0 {
                    v.major == 0 && v.minor == req.minor
                } else {
                    // req is 0.0.x -- only the exact patch is compatible (patch is the
                    // breaking boundary at 0.0.x, per Cargo's own caret rule).
                    v.major == 0 && v.minor == 0 && v.patch == req.patch
                }
            }
            VersionReq::Tilde(req, major_only) => {
                if *major_only {
                    v.major == req.major
                } else {
                    v >= req && v.major == req.major && v.minor == req.minor
                }
            }
        }
    }
}

impl std::fmt::Display for VersionReq {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            VersionReq::Exact(v) => write!(f, "={v}"),
            VersionReq::Caret(v) => write!(f, "^{v}"),
            VersionReq::Tilde(v, _) => write!(f, "~{v}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_and_partial_versions() {
        assert_eq!(Version::parse("1.2.3").unwrap(), Version { major: 1, minor: 2, patch: 3 });
        assert_eq!(Version::parse("1.2").unwrap(), Version { major: 1, minor: 2, patch: 0 });
        assert_eq!(Version::parse("1").unwrap(), Version { major: 1, minor: 0, patch: 0 });
    }

    #[test]
    fn rejects_malformed_versions() {
        assert!(Version::parse("1.2.3.4").is_err());
        assert!(Version::parse("1.x.3").is_err());
        assert!(Version::parse("").is_err());
    }

    #[test]
    fn parses_req_forms_and_defaults_bare_to_caret() {
        assert_eq!(VersionReq::parse("=1.2.3").unwrap(), VersionReq::Exact(Version::parse("1.2.3").unwrap()));
        assert_eq!(VersionReq::parse("^1.2.3").unwrap(), VersionReq::Caret(Version::parse("1.2.3").unwrap()));
        assert_eq!(VersionReq::parse("~1.2.3").unwrap(), VersionReq::Tilde(Version::parse("1.2.3").unwrap(), false));
        assert_eq!(VersionReq::parse("1.2.3").unwrap(), VersionReq::Caret(Version::parse("1.2.3").unwrap()));
    }

    #[test]
    fn caret_normal_major_allows_minor_and_patch_bumps_only() {
        let req = VersionReq::parse("^1.2.3").unwrap();
        assert!(req.matches(&Version::parse("1.2.3").unwrap()));
        assert!(req.matches(&Version::parse("1.2.4").unwrap()));
        assert!(req.matches(&Version::parse("1.9.0").unwrap()));
        assert!(!req.matches(&Version::parse("1.2.2").unwrap()), "below the minimum must not match");
        assert!(!req.matches(&Version::parse("2.0.0").unwrap()), "a major bump must not match");
    }

    #[test]
    fn caret_zero_major_treats_minor_as_the_breaking_boundary() {
        let req = VersionReq::parse("^0.2.3").unwrap();
        assert!(req.matches(&Version::parse("0.2.3").unwrap()));
        assert!(req.matches(&Version::parse("0.2.9").unwrap()));
        assert!(!req.matches(&Version::parse("0.3.0").unwrap()));
        assert!(!req.matches(&Version::parse("1.0.0").unwrap()));
    }

    #[test]
    fn caret_zero_zero_major_minor_treats_patch_as_the_breaking_boundary() {
        let req = VersionReq::parse("^0.0.3").unwrap();
        assert!(req.matches(&Version::parse("0.0.3").unwrap()));
        assert!(!req.matches(&Version::parse("0.0.4").unwrap()));
        assert!(!req.matches(&Version::parse("0.1.0").unwrap()));
    }

    #[test]
    fn tilde_allows_patch_bumps_only() {
        let req = VersionReq::parse("~1.2.3").unwrap();
        assert!(req.matches(&Version::parse("1.2.3").unwrap()));
        assert!(req.matches(&Version::parse("1.2.9").unwrap()));
        assert!(!req.matches(&Version::parse("1.3.0").unwrap()));
        assert!(!req.matches(&Version::parse("1.2.2").unwrap()));
    }

    #[test]
    fn tilde_with_only_major_and_minor_allows_patch_bumps_only() {
        // `~1.2` (no patch given) behaves the same as `~1.2.0` — patch may vary, minor may not.
        let req = VersionReq::parse("~1.2").unwrap();
        assert!(req.matches(&Version::parse("1.2.0").unwrap()));
        assert!(req.matches(&Version::parse("1.2.9").unwrap()));
        assert!(!req.matches(&Version::parse("1.3.0").unwrap()));
    }

    #[test]
    fn tilde_with_only_major_allows_any_minor_patch() {
        let req = VersionReq::parse("~1").unwrap();
        assert!(req.matches(&Version::parse("1.0.0").unwrap()));
        assert!(req.matches(&Version::parse("1.9.9").unwrap()));
        assert!(!req.matches(&Version::parse("2.0.0").unwrap()));
    }

    #[test]
    fn exact_matches_only_the_exact_version() {
        let req = VersionReq::parse("=1.2.3").unwrap();
        assert!(req.matches(&Version::parse("1.2.3").unwrap()));
        assert!(!req.matches(&Version::parse("1.2.4").unwrap()));
    }
}
