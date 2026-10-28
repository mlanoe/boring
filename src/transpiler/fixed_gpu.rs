// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
//! Binary fixed-point types (`fixedN_M` / `ufixedN_M`) in C-like GPU device code
//! (CUDA C, HIP) — docs/fixed-point-types.md, "GPU backends".
//!
//! A fixed-point value lowers to its raw integer. `+ - %` and comparisons are
//! plain integer ops; `*` is `(wide)a * b / (1 << M)` and `/` is
//! `((wide)a << M) / b`, with `/` (truncation toward zero), never `>>`. `wide` is
//! `long long` for `W <= 32` and `__int128` for `W = 64`. A 128-bit format is a
//! compile error. Overflow wraps (the interpreter errors); division by zero is
//! undefined, as for every integer on a GPU.
//!
//! The device emitters (`cuda::device`, `rocm::device`) keep a [`FixedScope`] of
//! the names whose type is fixed-point (kernel fields, parameters, `let`s) and
//! call the pure string formatters below.

use crate::ast::*;
use crate::fixed::FixedFmt;
use crate::transpiler::TranspileError;
use std::collections::HashMap;

/// The fixed-point format a type names, looking through `mut`, ownership
/// qualifiers and array/optional wrappers (an array of fixed is tracked by its
/// element format).
pub(crate) fn fmt_of_type(ty: &Type) -> Option<FixedFmt> {
    match ty {
        Type::Fixed(f) => Some(*f),
        Type::Named(n) => FixedFmt::from_name(n),
        Type::Mut(t) | Type::Qualified(t, _) | Type::Optional(t)
        | Type::Array(t) | Type::ArrayN(t, _) | Type::ArrayNExpr(t, _) | Type::LabeledArray(t, _) => fmt_of_type(t),
        _ => None,
    }
}

/// C type of the raw storage integer.
pub(crate) fn raw_c_type(f: &FixedFmt) -> String {
    let (u, w) = (if f.signed { "" } else { "u" }, f.width());
    match w {
        8 | 16 | 32 | 64 => format!("{u}int{w}_t"),
        _ => if f.signed { "__int128".into() } else { "unsigned __int128".into() },
    }
}

/// Intermediate type wide enough for the `2W`-bit product / shifted dividend.
fn wide(f: &FixedFmt) -> &'static str {
    match (f.width() <= 32, f.signed) {
        (true, true) => "long long",
        (true, false) => "unsigned long long",
        (false, true) => "__int128",
        (false, false) => "unsigned __int128",
    }
}

fn pow2(f: &FixedFmt, m: u32) -> String {
    match (f.width() <= 32, f.signed) {
        (true, true) => format!("(1LL << {m})"),
        (true, false) => format!("(1ULL << {m})"),
        _ => format!("(({})1 << {m})", wide(f)),
    }
}

/// Reports every fixed-point type in `ty` that no GPU target can lower (128-bit).
pub(crate) fn check_type(ty: &Type, line: usize, col: usize, errs: &mut Vec<TranspileError>) {
    if let Some(f) = fmt_of_type(ty) {
        if f.width() > 64 {
            errs.push(TranspileError::at(format!(
                "{} is not supported in GPU kernel code: 128-bit fixed-point formats need a 256-bit \
                 intermediate for `*` and `/` (docs/fixed-point-types.md, GPU backends)", f.name()), line, col));
        }
    }
}

pub(crate) fn unsupported(what: &str, line: usize, col: usize, errs: &mut Vec<TranspileError>) {
    errs.push(TranspileError::at(format!(
        "`{what}` on a fixed-point value is not supported in GPU kernel code — only `+ - * / %`, comparisons, \
         `as` casts, `.raw`, `fromRaw`, `abs`, `min`, `max` and `floor` are lowered"), line, col));
}

#[derive(Default, Clone)]
pub(crate) struct FixedScope {
    vars: HashMap<String, FixedFmt>,
}

impl FixedScope {
    pub(crate) fn clear(&mut self) { self.vars.clear(); }

    /// Records (or, for a non-fixed type, forgets) `name`'s format.
    pub(crate) fn declare(&mut self, name: &str, fmt: Option<FixedFmt>) {
        match fmt {
            Some(f) => { self.vars.insert(name.to_string(), f); }
            None => { self.vars.remove(name); }
        }
    }

    /// Static format of an expression, when fixed-point. Arithmetic takes the
    /// format of whichever operand has one.
    pub(crate) fn fmt_of(&self, e: &Expr) -> Option<FixedFmt> {
        match &e.kind {
            ExprKind::Var(v) => self.vars.get(v).copied(),
            ExprKind::Index(a, _) => match &a.kind { ExprKind::Var(v) => self.vars.get(v).copied(), _ => None },
            ExprKind::Cast(_, ty) => fmt_of_type(ty),
            ExprKind::BinOp(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem, l, r) =>
                self.fmt_of(l).or_else(|| self.fmt_of(r)),
            ExprKind::UnaryOp(UnaryOp::Neg, i) => self.fmt_of(i),
            ExprKind::MethodCall(o, m, _) => match (&o.kind, m.as_str()) {
                (ExprKind::Var(t), "fromRaw") => FixedFmt::from_name(t),
                (_, "abs" | "min" | "max" | "floor") => self.fmt_of(o),
                _ => None,
            },
            _ => None,
        }
    }
}

/// `Some(raw)` for an untyped numeric literal (optionally negated); the inner
/// option is "fits `fmt`".
fn literal_raw(e: &Expr, f: &FixedFmt) -> Option<Option<i128>> {
    match &e.kind {
        ExprKind::Int(n) => Some(f.from_int(*n as i128)),
        ExprKind::Float(x) => Some(f.from_f64_literal(*x)),
        ExprKind::UnaryOp(UnaryOp::Neg, i) => match &i.kind {
            ExprKind::Int(n) => Some(f.from_int(-(*n as i128))),
            ExprKind::Float(x) => Some(f.from_f64_literal(-*x)),
            _ => None,
        },
        _ => None,
    }
}

/// C constant of raw value `raw` in `f`'s storage type.
fn raw_const(f: &FixedFmt, raw: i128) -> String {
    let t = raw_c_type(f);
    let body = if raw == i64::MIN as i128 {
        "(-9223372036854775807LL-1)".to_string()
    } else if raw < 0 {
        format!("({raw})")
    } else if f.width() == 64 {
        format!("{raw}{}", if f.signed { "LL" } else { "ULL" })
    } else {
        raw.to_string()
    };
    format!("(({t}){body})")
}

/// An untyped literal folded to a `fmt` constant; `None` when `e` is not a literal.
/// Out of range is a compile error (and a `0` placeholder).
pub(crate) fn literal(e: &Expr, f: &FixedFmt, errs: &mut Vec<TranspileError>) -> Option<String> {
    Some(match literal_raw(e, f)? {
        Some(raw) => raw_const(f, raw),
        None => {
            errs.push(TranspileError::at(format!(
                "literal out of range for {} (range [{}, {}])",
                f.name(), f.format_exact(f.min_raw()), f.format_exact(f.max_raw())), e.line, e.col));
            raw_const(f, 0)
        }
    })
}

/// `l op r` on two emitted fixed-point operands; `None` for operators that are
/// not arithmetic/comparison (`&&`, `|`, ...).
pub(crate) fn binop(op: &BinOp, f: &FixedFmt, l: &str, r: &str) -> Option<String> {
    let t = raw_c_type(f);
    let w = wide(f);
    Some(match op {
        BinOp::Add => format!("(({t})({l} + {r}))"),
        BinOp::Sub => format!("(({t})({l} - {r}))"),
        BinOp::Rem => format!("(({t})({l} % {r}))"),
        BinOp::Mul => format!("(({t})(({w}){l} * ({w}){r} / {}))", pow2(f, f.m)),
        BinOp::Div => format!("(({t})((({w}){l} << {}) / ({w}){r}))", f.m),
        BinOp::Eq => format!("({l} == {r})"),
        BinOp::NotEq => format!("({l} != {r})"),
        BinOp::Lt => format!("({l} < {r})"),
        BinOp::Gt => format!("({l} > {r})"),
        BinOp::LtEq => format!("({l} <= {r})"),
        BinOp::GtEq => format!("({l} >= {r})"),
        _ => return None,
    })
}

pub(crate) fn neg(f: &FixedFmt, v: &str) -> String { format!("(({})(-{v}))", raw_c_type(f)) }

/// `v as <target>` where the source is fixed-point `src` and/or the target is.
/// `inner` is the already-emitted operand, `target_c` the C type of a non-fixed
/// target. `None` when neither side is fixed.
pub(crate) fn cast(src: Option<&FixedFmt>, to: Option<&FixedFmt>, inner: &str, target_c: &str, target_is_float: bool) -> Option<String> {
    match (src, to) {
        (Some(s), Some(t)) => {
            let (ts, w) = (raw_c_type(t), if s.width() > 32 || t.width() > 32 { "__int128" } else { "long long" });
            Some(if t.m > s.m {
                format!("(({ts})(({w}){inner} << {}))", t.m - s.m)
            } else if t.m < s.m {
                format!("(({ts})(({w}){inner} / (({w})1 << {})))", s.m - t.m)
            } else {
                format!("(({ts}){inner})")
            })
        }
        (None, Some(t)) => {
            let ts = raw_c_type(t);
            Some(if target_is_float {
                format!("(({ts})((double){inner} * {}.0))", 1u128 << t.m)
            } else {
                format!("(({ts})(({}){inner} << {}))", wide(t), t.m)
            })
        }
        (Some(s), None) => Some(if target_is_float {
            format!("(({target_c})((double){inner} / {}.0))", 1u128 << s.m)
        } else {
            format!("(({target_c})(({}){inner} / {}))", wide(s), pow2(s, s.m))
        }),
        (None, None) => None,
    }
}

/// Members lowered in device code: `abs`, `min`, `max`, `floor`, `fromRaw`.
/// `Err(())` means "a fixed-point member that GPU code does not support".
pub(crate) fn method(f: &FixedFmt, method: &str, recv: &str, args: &[String]) -> Result<String, ()> {
    let t = raw_c_type(f);
    Ok(match (method, args) {
        ("abs", []) => format!("(({t})(({recv}) < 0 ? -({recv}) : ({recv})))"),
        ("min", [a]) => format!("(({recv}) < ({a}) ? ({recv}) : ({a}))"),
        ("max", [a]) => format!("(({recv}) > ({a}) ? ({recv}) : ({a}))"),
        ("floor", []) => format!("(({t})((({recv}) >> {}) << {}))", f.m, f.m),
        _ => return Err(()),
    })
}
