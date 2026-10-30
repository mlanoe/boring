// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
//! Binary fixed-point types (`fixedN_M` / `ufixedN_M`) in WGSL device code
//! (docs/fixed-point-types.md, "GPU backends" / "wgpu").
//!
//! A fixed-point value lowers to its raw integer: `i32` (signed) or `u32` (unsigned). WGSL has
//! no 8/16-bit scalars, so a narrower raw is kept sign-/zero-extended in the 32-bit register and
//! re-narrowed after every operation that can leave its range (`extractBits(x, 0u, W)`).
//! Buffers of 8/16-bit raws reuse the packed `array<u32>` support (`PackedByteKind`).
//!
//! * `W <= 16`: `*` and `/` stay inside 32 bits (`|ra*rb| <= 2^30`, `ra << M < 2^32`), so they are
//!   plain `i32`/`u32` ops with `/` truncating toward zero.
//! * `W = 32`: the 64-bit product / shifted dividend is emulated with `u32` limb helpers
//!   (`bp_fx_*`, emitted once into the WGSL module only when used) on the magnitudes, with the
//!   sign applied afterwards (truncation toward zero, never an arithmetic shift).
//! * `W = 64` / `128`: compile error on this target.
//!
//! Overflow wraps and division by zero is undefined, as for any GPU integer op.

use super::*;
use crate::fixed::FixedFmt;

/// The fixed-point format a scalar type names (no array/optional look-through).
pub(super) fn fixed_fmt_of_type(ty: &Type) -> Option<FixedFmt> {
    match ty {
        Type::Fixed(f) => Some(*f),
        Type::Named(n) => FixedFmt::from_name(n),
        Type::Mut(t) | Type::Qualified(t, _) => fixed_fmt_of_type(t),
        _ => None,
    }
}

/// Element type of an array-shaped type.
fn array_elem_type(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Array(t) | Type::ArrayN(t, _) | Type::ArrayNExpr(t, _) | Type::LabeledArray(t, _) => Some(t),
        Type::Mut(t) | Type::Qualified(t, _) => array_elem_type(t),
        _ => ty.as_labeled_array().map(|(t, _)| t),
    }
}

fn is_float_ty(t: &Type) -> bool {
    match t.without_mut() {
        Type::Float16 | Type::Float32 | Type::Float64 => true,
        Type::Named(n) => matches!(n.as_str(), "float" | "float16" | "float32" | "float64" | "f16" | "f32" | "f64"),
        Type::Qualified(inner, _) => is_float_ty(inner),
        _ => false,
    }
}

/// WGSL storage type of the raw integer.
pub(super) fn raw_t(f: &FixedFmt) -> &'static str { if f.signed { "i32" } else { "u32" } }

fn suffix(f: &FixedFmt) -> &'static str { if f.signed { "" } else { "u" } }

/// `2^k` as a typed WGSL literal of `f`'s raw type (`k < 32`, or `k == 32` for unsigned: `None`).
fn pow2_lit(f: &FixedFmt, k: u32) -> Option<String> {
    if k > 31 { return None; }
    Some(format!("{}{}", 1u64 << k, suffix(f)))
}

/// A typed constant of `f`'s raw type holding the 32-bit pattern `bits`.
fn bits_const(f: &FixedFmt, bits: u32) -> String {
    if f.signed { format!("i32({})", bits as i32) } else { format!("{bits}u") }
}

/// Re-narrows a raw expression to `f`'s width (sign-/zero-extending); a no-op for 32-bit formats.
fn wrap(f: &FixedFmt, e: String) -> String {
    if f.width() >= 32 { e } else { format!("extractBits({e}, 0u, {}u)", f.width()) }
}

/// Bit-preserving conversion of a raw expression of format `from` to `to`'s raw type.
fn reinterpret(from: &FixedFmt, to: &FixedFmt, e: String) -> String {
    if from.signed == to.signed { e } else { format!("{}({e})", raw_t(to)) }
}

/// `e << k` on a raw value (all-zero when `k >= 32`).
fn shl(f: &FixedFmt, e: &str, k: u32) -> String {
    if k == 0 { e.to_string() }
    else if k >= 32 { bits_const(f, 0) }
    else { format!("({e} << {k}u)") }
}

const HELPER_UMUL64: &str = "fn bp_fx_umul64(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu; let a1 = a >> 16u;
    let b0 = b & 0xffffu; let b1 = b >> 16u;
    let p00 = a0 * b0; let p01 = a0 * b1; let p10 = a1 * b0; let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    let lo = (p00 & 0xffffu) | (mid << 16u);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}
fn bp_fx_shr64(lo: u32, hi: u32, m: u32) -> u32 {
    if (m == 0u) { return lo; }
    if (m >= 32u) { return hi; }
    return (lo >> m) | (hi << (32u - m));
}
";
const HELPER_MUL_U32: &str = "fn bp_fx_mul_u32(a: u32, b: u32, m: u32) -> u32 {
    let p = bp_fx_umul64(a, b);
    return bp_fx_shr64(p.x, p.y, m);
}
";
const HELPER_MUL_I32: &str = "fn bp_fx_mul_i32(a: i32, b: i32, m: u32) -> i32 {
    let neg = (a < 0) != (b < 0);
    let r = bp_fx_mul_u32(bitcast<u32>(select(a, -a, a < 0)), bitcast<u32>(select(b, -b, b < 0)), m);
    return bitcast<i32>(select(r, 0u - r, neg));
}
";
const HELPER_UDIV64: &str = "fn bp_fx_udiv64(nlo: u32, nhi: u32, d: u32) -> u32 {
    var r: u32 = 0u;
    var q: u32 = 0u;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        let top = r >> 31u;
        let src = select(nlo, nhi, i < 32u);
        r = (r << 1u) | ((src >> (31u - (i & 31u))) & 1u);
        q = q << 1u;
        if (top == 1u || r >= d) { r = r - d; q = q | 1u; }
    }
    return q;
}
";
const HELPER_DIV_U32: &str = "fn bp_fx_div_u32(a: u32, b: u32, m: u32) -> u32 {
    var nlo: u32 = a;
    var nhi: u32 = 0u;
    if (m >= 32u) { nlo = 0u; nhi = a; }
    else if (m > 0u) { nlo = a << m; nhi = a >> (32u - m); }
    return bp_fx_udiv64(nlo, nhi, b);
}
";
const HELPER_DIV_I32: &str = "fn bp_fx_div_i32(a: i32, b: i32, m: u32) -> i32 {
    let neg = (a < 0) != (b < 0);
    let q = bp_fx_div_u32(bitcast<u32>(select(a, -a, a < 0)), bitcast<u32>(select(b, -b, b < 0)), m);
    return bitcast<i32>(select(q, 0u - q, neg));
}
";
const HELPER_TDIV_I32: &str = "fn bp_fx_tdiv_i32(v: i32, k: u32) -> i32 {
    let u = bitcast<u32>(select(v, -v, v < 0));
    let q = select(u >> k, 0u, k >= 32u);
    return bitcast<i32>(select(q, 0u - q, v < 0));
}
";

/// Helper sources in dependency order: (name, text, depends on).
const HELPERS: &[(&str, &str, &[&str])] = &[
    ("umul64", HELPER_UMUL64, &[]),
    ("mul_u32", HELPER_MUL_U32, &["umul64"]),
    ("mul_i32", HELPER_MUL_I32, &["mul_u32"]),
    ("udiv64", HELPER_UDIV64, &[]),
    ("div_u32", HELPER_DIV_U32, &["udiv64"]),
    ("div_i32", HELPER_DIV_I32, &["div_u32"]),
    ("tdiv_i32", HELPER_TDIV_I32, &[]),
];

impl DeviceEmitter {
    /// The WGSL source of every `bp_fx_*` helper used so far (dependency order).
    pub(super) fn fixed_helpers_wgsl(&self) -> String {
        let mut used: std::collections::BTreeSet<&str> = self.fixed_helpers.iter().copied().collect();
        loop {
            let before = used.len();
            for (name, _, deps) in HELPERS {
                if used.contains(name) { used.extend(deps.iter().copied()); }
            }
            if used.len() == before { break; }
        }
        HELPERS.iter().filter(|(n, _, _)| used.contains(n)).map(|(_, t, _)| *t).collect()
    }

    fn use_fixed_helper(&mut self, name: &'static str) { self.fixed_helpers.insert(name); }

    // ── type tracking ───────────────────────────────────────────────────────────────

    fn local_or_field_type(&self, name: &str) -> Option<Type> {
        self.locals.get(name).cloned()
            .or_else(|| self.current_fields.iter().find(|f| f.name == name).map(|f| f.ty.clone()))
    }

    /// Best-effort static type of a scalar expression: just enough to find a fixed-point format
    /// (or tell a float from an integer) for the lowering below.
    pub(super) fn ty_of(&self, e: &Expr) -> Option<Type> {
        match &e.kind {
            ExprKind::Var(n) => self.local_or_field_type(n),
            ExprKind::Field(obj, n) if matches!(&obj.kind, ExprKind::Var(v) if v == "self") => self.local_or_field_type(n),
            ExprKind::Field(obj, n) if matches!(n.as_str(), "epsilon" | "min" | "max") => match &obj.kind {
                ExprKind::Var(v) if self.local_or_field_type(v).is_none() => FixedFmt::from_name(v).map(Type::Fixed),
                _ => None,
            },
            ExprKind::Index(arr, _) => self.ty_of(arr).and_then(|t| array_elem_type(&t).cloned()),
            ExprKind::Cast(_, ty) => Some(ty.clone()),
            ExprKind::Int(_) => Some(Type::Int),
            ExprKind::Float(_) => Some(Type::Float64),
            ExprKind::BinOp(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem, l, r) =>
                self.ty_of(l).filter(|t| fixed_fmt_of_type(t).is_some() || is_float_ty(t))
                    .or_else(|| self.ty_of(r).filter(|t| fixed_fmt_of_type(t).is_some() || is_float_ty(t)))
                    .or_else(|| self.ty_of(l)),
            ExprKind::UnaryOp(UnaryOp::Neg, inner) => self.ty_of(inner),
            ExprKind::MethodCall(obj, m, _) if matches!(m.as_str(), "floor" | "ceil" | "trunc" | "fract" | "abs" | "min" | "max")
                && self.fixed_fmt_of(obj).is_some() => self.ty_of(obj),
            ExprKind::MethodCall(obj, m, _) if m == "fromRaw" => match &obj.kind {
                ExprKind::Var(v) if self.local_or_field_type(v).is_none() => FixedFmt::from_name(v).map(Type::Fixed),
                _ => None,
            },
            ExprKind::Call(callee, _) => match &callee.kind {
                ExprKind::Var(n) => self.fn_ret_fmts.get(n).copied().flatten().map(Type::Fixed),
                _ => None,
            },
            ExprKind::MethodCall(obj, m, _) if matches!(&obj.kind, ExprKind::Var(v) if v == "self") =>
                self.fn_ret_fmts.get(&format!("self::{m}")).copied().flatten().map(Type::Fixed),
            _ => None,
        }
    }

    pub(super) fn fixed_fmt_of(&self, e: &Expr) -> Option<FixedFmt> {
        self.ty_of(e).as_ref().and_then(fixed_fmt_of_type)
    }

    /// A format this backend can do; anything else is a compile error (never a silent narrowing).
    fn fixed_supported(&mut self, f: &FixedFmt, e: &Expr) {
        self.check_fixed_format(f, e.line, e.col);
    }

    pub(super) fn check_fixed_format(&mut self, f: &FixedFmt, line: usize, col: usize) {
        if f.width() > 32 {
            let msg = format!(
                "`{}` is not supported on --target wgpu: WGSL has no 64-bit integers and the 64-bit \
                 emulation is only implemented for formats of at most 32 bits (use e.g. `fixed16_16`)",
                f.name());
            if !self.errors.iter().any(|e| e.message == msg && e.line == line && e.col == col) {
                self.errors.push(TranspileError::at(msg, line, col));
            }
        }
    }

    // ── literals / operands ─────────────────────────────────────────────────────────

    /// An untyped numeric literal (optionally negated) as the raw integer of `f`.
    pub(super) fn fixed_literal(&mut self, e: &Expr, f: &FixedFmt) -> Option<String> {
        let raw = match &e.kind {
            ExprKind::Int(n) => f.from_int(*n as i128),
            ExprKind::Float(x) => f.from_f64_literal(*x),
            ExprKind::UnaryOp(UnaryOp::Neg, inner) => match &inner.kind {
                ExprKind::Int(n) => f.from_int(-(*n as i128)),
                ExprKind::Float(x) => f.from_f64_literal(-*x),
                _ => return None,
            },
            _ => return None,
        };
        Some(match raw {
            Some(r) if f.width() > 32 => { let _ = r; "0".to_string() }
            Some(r) => if f.signed { format!("i32({r})") } else { format!("{r}u") },
            None => {
                self.errors.push(TranspileError::at(
                    format!("literal out of range for {} (range [{}, {}])",
                        f.name(), f.format_exact(f.min_raw()), f.format_exact(f.max_raw())),
                    e.line, e.col));
                bits_const(f, 0)
            }
        })
    }

    /// Operand of a fixed-point operation: a bare literal folds to the raw integer.
    pub(super) fn fixed_operand(&mut self, e: &Expr, f: &FixedFmt) -> String {
        match self.fixed_literal(e, f) {
            Some(s) => s,
            None => self.expr(e),
        }
    }

    // ── operators ───────────────────────────────────────────────────────────────────

    pub(super) fn fixed_binop_fmt(&self, op: &BinOp, l: &Expr, r: &Expr) -> Option<FixedFmt> {
        if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem
            | BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq) {
            return None;
        }
        self.fixed_fmt_of(l).or_else(|| self.fixed_fmt_of(r))
    }

    pub(super) fn fixed_binop(&mut self, op: &BinOp, l: &Expr, r: &Expr, f: &FixedFmt, e: &Expr) -> String {
        self.fixed_supported(f, e);
        let a = self.fixed_operand(l, f);
        let b = self.fixed_operand(r, f);
        let m = f.m;
        match op {
            BinOp::Add => wrap(f, format!("({a} + {b})")),
            BinOp::Sub => wrap(f, format!("({a} - {b})")),
            BinOp::Rem => wrap(f, format!("({a} % {b})")),
            BinOp::Mul if f.width() >= 32 => {
                let h = if f.signed { "mul_i32" } else { "mul_u32" };
                self.use_fixed_helper(h);
                format!("bp_fx_{h}({a}, {b}, {m}u)")
            }
            // `/`, never `>>`: the quotient truncates toward zero, an arithmetic shift would floor.
            BinOp::Mul => match pow2_lit(f, m) {
                Some(p) => wrap(f, format!("(({a} * {b}) / {p})")),
                None => unreachable!("W < 32 implies M < 32"),
            },
            BinOp::Div if f.width() >= 32 => {
                let h = if f.signed { "div_i32" } else { "div_u32" };
                self.use_fixed_helper(h);
                format!("bp_fx_{h}({a}, {b}, {m}u)")
            }
            BinOp::Div => wrap(f, format!("({} / {b})", shl(f, &format!("({a})"), m))),
            cmp => format!("({a} {} {b})", binop_wgsl(cmp)),
        }
    }

    pub(super) fn fixed_neg(&mut self, operand: &Expr, f: &FixedFmt, e: &Expr) -> String {
        self.fixed_supported(f, e);
        let v = self.expr(operand);
        if f.signed { wrap(f, format!("(-{v})")) } else { wrap(f, format!("(0u - {v})")) }
    }

    /// `x.raw`, `fixed16_16.epsilon` / `.min` / `.max`.
    pub(super) fn fixed_field(&mut self, obj: &Expr, field: &str, e: &Expr) -> Option<String> {
        if let ExprKind::Var(v) = &obj.kind {
            if let Some(f) = FixedFmt::from_name(v) {
                if self.local_or_field_type(v).is_some() { return None; }
                self.fixed_supported(&f, e);
                let raw = match field {
                    "epsilon" => 1,
                    "min" => f.min_raw(),
                    "max" => f.max_raw(),
                    _ => return None,
                };
                if f.width() > 32 { return Some("0".into()); }
                return Some(if f.signed { format!("i32({raw})") } else { format!("{raw}u") });
            }
        }
        if field == "raw" {
            if let Some(f) = self.fixed_fmt_of(obj) {
                self.fixed_supported(&f, e);
                // `.raw` is an `int` (32-bit on this target); an unsigned 32-bit raw above 2^31 wraps.
                let o = self.expr(obj);
                return Some(if f.signed { o } else { format!("i32({o})") });
            }
        }
        None
    }

    pub(super) fn fixed_method_applies(&self, obj: &Expr) -> bool {
        if let ExprKind::Var(v) = &obj.kind {
            if FixedFmt::from_name(v).is_some() && self.local_or_field_type(v).is_none() { return true; }
        }
        self.fixed_fmt_of(obj).is_some()
    }

    /// `x` truncated toward zero to an integer multiple of `2^M`'s quotient: `trunc(x / 2^M)` as raw-typed.
    fn fixed_trunc_div(&mut self, f: &FixedFmt, v: &str, k: u32) -> String {
        if k == 0 { return v.to_string(); }
        if f.signed {
            if f.width() >= 32 {
                self.use_fixed_helper("tdiv_i32");
                format!("bp_fx_tdiv_i32({v}, {k}u)")
            } else {
                format!("({v} / {})", 1u64 << k)
            }
        } else if k >= 32 {
            "0u".to_string()
        } else {
            format!("({v} >> {k}u)")
        }
    }

    pub(super) fn fixed_method(&mut self, obj: &Expr, method: &str, args: &[Arg], e: &Expr) -> String {
        // `fixed16_16.fromRaw(n)`
        if let ExprKind::Var(v) = &obj.kind {
            if let Some(f) = FixedFmt::from_name(v) {
                if self.local_or_field_type(v).is_none() {
                    self.fixed_supported(&f, e);
                    if let ("fromRaw", [a]) = (method, args) {
                        let n = self.expr(&a.value);
                        return wrap(&f, format!("{}({n})", raw_t(&f)));
                    }
                    return self.fixed_gpu_error(&f, method, e);
                }
            }
        }
        let f = self.fixed_fmt_of(obj).unwrap();
        self.fixed_supported(&f, e);
        let o = self.expr(obj);
        let m = f.m;
        let mask: u32 = if m >= 32 { u32::MAX } else { (1u32 << m) - 1 };
        let inv = bits_const(&f, !mask);
        let maskc = bits_const(&f, mask);
        match (method, args) {
            ("floor", []) => format!("({o} & {inv})"),
            ("ceil", []) => wrap(&f, format!("(({o} + {maskc}) & {inv})")),
            ("trunc", []) => {
                let q = self.fixed_trunc_div(&f, &o, m);
                shl(&f, &q, m)
            }
            ("fract", []) => {
                let q = self.fixed_trunc_div(&f, &o, m);
                wrap(&f, format!("({o} - {})", shl(&f, &q, m)))
            }
            ("abs", []) if f.signed => wrap(&f, format!("abs({o})")),
            ("abs", []) => o,
            ("min" | "max", [a]) => {
                let b = self.fixed_operand(&a.value, &f);
                format!("{method}({o}, {b})")
            }
            _ => self.fixed_gpu_error(&f, method, e),
        }
    }

    fn fixed_gpu_error(&mut self, f: &FixedFmt, method: &str, e: &Expr) -> String {
        self.errors.push(TranspileError::at(
            format!("`{}.{}(..)` is not supported on --target wgpu: fixed-point kernels support \
                     `+ - * / %`, comparisons, `as` casts, `.raw`, `fromRaw`, `floor`, `ceil`, `trunc`, `fract`, \
                     `abs`, `min` and `max`; math functions (`sqrt`, `sin`, ...), `round`, `signum`, `clamp` and \
                     the `wrapping*`/`saturating*`/`checked*`/`mulRound`/`divRound` variants are CPU-only", f.name(), method),
            e.line, e.col));
        "0".to_string()
    }

    // ── casts ───────────────────────────────────────────────────────────────────────

    /// `x as T` where `x` or `T` is fixed-point.
    pub(super) fn fixed_cast(&mut self, inner: &Expr, ty: &Type, e: &Expr) -> String {
        let src = self.fixed_fmt_of(inner);
        if let Some(to) = fixed_fmt_of_type(ty) {
            self.fixed_supported(&to, e);
            if let Some(lit) = self.fixed_literal(inner, &to) { return lit; }
            let v = self.expr(inner);
            if let Some(from) = src {
                self.fixed_supported(&from, e);
                let wide = if to.m >= from.m {
                    // Left shift (exact modulo 2^32, which the re-narrowing below makes exact modulo 2^W).
                    shl(&from, &v, to.m - from.m)
                } else {
                    self.fixed_trunc_div(&from, &v, from.m - to.m)
                };
                return wrap(&to, reinterpret(&from, &to, wide));
            }
            let is_float = self.ty_of(inner).as_ref().map(is_float_ty).unwrap_or(false)
                || self.wgsl_is_float_expr(inner);
            let scale = 2f64.powi(to.m as i32);
            return if is_float {
                // float -> fixed truncates toward zero (WGSL's float -> int conversion does too).
                wrap(&to, format!("{}(f32({v}) * {scale:.1})", raw_t(&to)))
            } else {
                wrap(&to, shl(&to, &format!("{}({v})", raw_t(&to)), to.m))
            };
        }
        // fixed -> int / float
        let from = src.unwrap();
        self.fixed_supported(&from, e);
        let v = self.expr(inner);
        let dst = wgsl_type(ty);
        if is_float_ty(ty) {
            let scale = 2f64.powi(from.m as i32);
            format!("{dst}(f32({v}) / {scale:.1})")
        } else {
            let q = self.fixed_trunc_div(&from, &v, from.m);
            format!("{dst}({q})")
        }
    }

    /// `kernel local array` initializer element: literals fold against the array's fixed element.
    pub(super) fn fixed_array_elem(&mut self, inner: &Type, v: &Expr) -> String {
        match fixed_fmt_of_type(inner) {
            Some(f) => self.fixed_operand(v, &f),
            None => self.expr(v),
        }
    }

    /// Rejects a device function whose parameter / return type is a format this backend cannot lower.
    pub(super) fn check_fixed_signature(&mut self, params: &[Param], ret: Option<&Type>, line: usize, col: usize) {
        for p in params {
            if let Some(f) = p.ty.as_ref().and_then(fixed_fmt_of_type) { self.check_fixed_format(&f, p.line, p.col); }
        }
        if let Some(f) = ret.and_then(fixed_fmt_of_type) { self.check_fixed_format(&f, line, col); }
    }

    /// Reports (once per field) fixed-point kernel fields this backend cannot lower.
    pub(super) fn check_fixed_fields(&mut self, decl: &KernelDecl) {
        for f in &decl.fields {
            let elem = array_elem_type(&f.ty).unwrap_or(&f.ty);
            let Some(fmt) = fixed_fmt_of_type(elem) else { continue };
            self.check_fixed_format(&fmt, f.line, f.col);
            if matches!(f.qual, GpuQual::ActorGlobal | GpuQual::ActorUnified) {
                self.errors.push(TranspileError::at(format!(
                    "kernel {}: `'actor` buffer field '{}' of fixed-point type {} is not supported on --target wgpu \
                     (WGSL atomics exist only for 32-bit `i32`/`u32`; there is no fixed-point atomic)",
                    decl.name, f.name, fmt.name()), f.line, f.col));
            }
        }
    }
}
