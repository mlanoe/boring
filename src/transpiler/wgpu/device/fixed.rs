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
//! * `W = 64`: the raw is a `vec2<u32>` (`.x` = low word, `.y` = high word, two's complement for
//!   both signednesses; a `[fixed32_32]` buffer is an `array<vec2<u32>>`, bit-identical to the
//!   host's `Vec<Fixed32_32>`). Add/sub/neg/compare/shift/`*`/`/`/`%` are `bp_fx_*64` helpers
//!   (emitted only when used): `*` builds the 128-bit product of the magnitudes from 32x32
//!   partial products and keeps bits `M..M+64`, `/` is a 64..127-step shift-subtract division of
//!   `|a| << M` by `|b|`, signs are applied last (truncation toward zero, wrapping final raw).
//!   A scalar 64-bit kernel parameter crosses the params struct as two `u32` slots
//!   (`name_lo` / `name_hi`). `.raw` is the low 32 bits (`int` is 32-bit on this target).
//!   A `'const` array of a 64-bit format (`[T, N]` or labeled) is `array<u32, 2N>` in the params block.
//! * `W = 128`: compile error on this target.
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
pub(super) fn raw_t(f: &FixedFmt) -> &'static str {
    if f.width() > 32 { "vec2<u32>" } else if f.signed { "i32" } else { "u32" }
}

/// Is the raw a `vec2<u32>` limb pair?
fn is64(f: &FixedFmt) -> bool { f.width() > 32 }

/// A 64-bit raw pattern as a WGSL constant (`i128` truncated to its low 64 bits).
fn lit64(raw: i128) -> String {
    let v = raw as u64;
    format!("vec2<u32>({}u, {}u)", v as u32, (v >> 32) as u32)
}

fn is_unsigned_int_ty(t: &Type) -> bool {
    match t.without_mut() {
        Type::Uint | Type::Uint8 | Type::Uint16 | Type::Uint32 | Type::Uint64 => true,
        Type::Named(n) => matches!(n.as_str(), "uint" | "uint8" | "uint16" | "uint32" | "uint64" | "u8" | "u16" | "u32" | "u64"),
        Type::Qualified(inner, _) => is_unsigned_int_ty(inner),
        _ => false,
    }
}

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

/// The raw zero of `f`.
fn zero_const(f: &FixedFmt) -> String {
    if is64(f) { "vec2<u32>(0u, 0u)".to_string() } else { bits_const(f, 0) }
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

const H_ADD64: &str = "fn bp_fx_add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}
";
const H_NEG64: &str = "fn bp_fx_neg64(a: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(0u - a.x, ~a.y + select(0u, 1u, a.x == 0u));
}
";
const H_SUB64: &str = "fn bp_fx_sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return bp_fx_add64(a, bp_fx_neg64(b)); }
";
const H_ABS64: &str = "fn bp_fx_abs64(a: vec2<u32>) -> vec2<u32> { return select(a, bp_fx_neg64(a), (a.y >> 31u) == 1u); }
";
const H_ULT64: &str = "fn bp_fx_ult64(a: vec2<u32>, b: vec2<u32>) -> bool { return a.y < b.y || (a.y == b.y && a.x < b.x); }
";
const H_SLT64: &str = "fn bp_fx_slt64(a: vec2<u32>, b: vec2<u32>) -> bool {
    let ah = bitcast<i32>(a.y);
    let bh = bitcast<i32>(b.y);
    return ah < bh || (ah == bh && a.x < b.x);
}
";
const H_UMIN64: &str = "fn bp_fx_umin64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return select(a, b, bp_fx_ult64(b, a)); }
";
const H_UMAX64: &str = "fn bp_fx_umax64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return select(a, b, bp_fx_ult64(a, b)); }
";
const H_SMIN64: &str = "fn bp_fx_smin64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return select(a, b, bp_fx_slt64(b, a)); }
";
const H_SMAX64: &str = "fn bp_fx_smax64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> { return select(a, b, bp_fx_slt64(a, b)); }
";
const H_SHL64: &str = "fn bp_fx_shl64(a: vec2<u32>, k: u32) -> vec2<u32> {
    if (k == 0u) { return a; }
    if (k >= 64u) { return vec2<u32>(0u, 0u); }
    if (k >= 32u) { return vec2<u32>(0u, a.x << (k - 32u)); }
    return vec2<u32>(a.x << k, (a.y << k) | (a.x >> (32u - k)));
}
";
const H_LSHR64: &str = "fn bp_fx_lshr64(a: vec2<u32>, k: u32) -> vec2<u32> {
    if (k == 0u) { return a; }
    if (k >= 64u) { return vec2<u32>(0u, 0u); }
    if (k >= 32u) { return vec2<u32>(a.y >> (k - 32u), 0u); }
    return vec2<u32>((a.x >> k) | (a.y << (32u - k)), a.y >> k);
}
";
const H_TDIV64: &str = "fn bp_fx_tdiv64(v: vec2<u32>, k: u32) -> vec2<u32> {
    let neg = (v.y >> 31u) == 1u;
    let q = bp_fx_lshr64(select(v, bp_fx_neg64(v), neg), k);
    return select(q, bp_fx_neg64(q), neg);
}
";
const H_UMULSHR64: &str = "fn bp_fx_umulshr64(a: vec2<u32>, b: vec2<u32>, m: u32) -> vec2<u32> {
    let p00 = bp_fx_umul64(a.x, b.x);
    let p01 = bp_fx_umul64(a.x, b.y);
    let p10 = bp_fx_umul64(a.y, b.x);
    let p11 = bp_fx_umul64(a.y, b.y);
    let r0 = p00.x;
    var r1 = p00.y;
    var c1 = 0u;
    var s = r1 + p01.x; c1 += select(0u, 1u, s < r1); r1 = s;
    s = r1 + p10.x; c1 += select(0u, 1u, s < r1); r1 = s;
    var r2 = p11.x;
    var c2 = 0u;
    s = r2 + p01.y; c2 += select(0u, 1u, s < r2); r2 = s;
    s = r2 + p10.y; c2 += select(0u, 1u, s < r2); r2 = s;
    s = r2 + c1; c2 += select(0u, 1u, s < r2); r2 = s;
    let r3 = p11.y + c2;
    if (m == 0u) { return vec2<u32>(r0, r1); }
    if (m < 32u) { return vec2<u32>((r0 >> m) | (r1 << (32u - m)), (r1 >> m) | (r2 << (32u - m))); }
    if (m == 32u) { return vec2<u32>(r1, r2); }
    if (m >= 64u) { return vec2<u32>(r2, r3); }
    let t = m - 32u;
    return vec2<u32>((r1 >> t) | (r2 << (32u - t)), (r2 >> t) | (r3 << (32u - t)));
}
";
const H_MUL_U64: &str = "fn bp_fx_mul_u64(a: vec2<u32>, b: vec2<u32>, m: u32) -> vec2<u32> { return bp_fx_umulshr64(a, b, m); }
";
const H_MUL_I64: &str = "fn bp_fx_mul_i64(a: vec2<u32>, b: vec2<u32>, m: u32) -> vec2<u32> {
    let na = (a.y >> 31u) == 1u;
    let nb = (b.y >> 31u) == 1u;
    let r = bp_fx_umulshr64(select(a, bp_fx_neg64(a), na), select(b, bp_fx_neg64(b), nb), m);
    return select(r, bp_fx_neg64(r), na != nb);
}
";
const H_UDIVSHL64: &str = "fn bp_fx_udivshl64(a: vec2<u32>, d: vec2<u32>, m: u32) -> vec2<u32> {
    var r = vec2<u32>(0u, 0u);
    var q = vec2<u32>(0u, 0u);
    let mi = i32(m);
    for (var i: i32 = 63 + mi; i >= 0; i = i - 1) {
        var bit = 0u;
        if (i >= mi) {
            let k = u32(i - mi);
            bit = (select(a.x, a.y, k >= 32u) >> (k & 31u)) & 1u;
        }
        let carry = r.y >> 31u;
        r = vec2<u32>((r.x << 1u) | bit, (r.y << 1u) | (r.x >> 31u));
        q = vec2<u32>(q.x << 1u, (q.y << 1u) | (q.x >> 31u));
        if (carry != 0u || !bp_fx_ult64(r, d)) { r = bp_fx_sub64(r, d); q.x = q.x | 1u; }
    }
    return q;
}
";
const H_DIV_U64: &str = "fn bp_fx_div_u64(a: vec2<u32>, b: vec2<u32>, m: u32) -> vec2<u32> { return bp_fx_udivshl64(a, b, m); }
";
const H_DIV_I64: &str = "fn bp_fx_div_i64(a: vec2<u32>, b: vec2<u32>, m: u32) -> vec2<u32> {
    let na = (a.y >> 31u) == 1u;
    let nb = (b.y >> 31u) == 1u;
    let q = bp_fx_udivshl64(select(a, bp_fx_neg64(a), na), select(b, bp_fx_neg64(b), nb), m);
    return select(q, bp_fx_neg64(q), na != nb);
}
";
const H_UREM64: &str = "fn bp_fx_urem64(a: vec2<u32>, d: vec2<u32>) -> vec2<u32> {
    var r = vec2<u32>(0u, 0u);
    for (var i: i32 = 63; i >= 0; i = i - 1) {
        let k = u32(i);
        let bit = (select(a.x, a.y, k >= 32u) >> (k & 31u)) & 1u;
        let carry = r.y >> 31u;
        r = vec2<u32>((r.x << 1u) | bit, (r.y << 1u) | (r.x >> 31u));
        if (carry != 0u || !bp_fx_ult64(r, d)) { r = bp_fx_sub64(r, d); }
    }
    return r;
}
";
const H_REM_I64: &str = "fn bp_fx_rem_i64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let na = (a.y >> 31u) == 1u;
    let nb = (b.y >> 31u) == 1u;
    let r = bp_fx_urem64(select(a, bp_fx_neg64(a), na), select(b, bp_fx_neg64(b), nb));
    return select(r, bp_fx_neg64(r), na);
}
";
const H_I32TO64: &str = "fn bp_fx_i32to64(x: i32) -> vec2<u32> { return vec2<u32>(bitcast<u32>(x), select(0u, 0xffffffffu, x < 0)); }
";
const H_F2RAW64: &str = "fn bp_fx_f2raw64(x: f32) -> vec2<u32> {
    let m = abs(x);
    let hi = u32(floor(m / 4294967296.0));
    let lo = u32(m - f32(hi) * 4294967296.0);
    let r = vec2<u32>(lo, hi);
    return select(r, bp_fx_neg64(r), x < 0.0);
}
";
const H_U64TOF32: &str = "fn bp_fx_u64tof32(a: vec2<u32>) -> f32 { return f32(a.y) * 4294967296.0 + f32(a.x); }
";
const H_I64TOF32: &str = "fn bp_fx_i64tof32(a: vec2<u32>) -> f32 {
    let n = (a.y >> 31u) == 1u;
    let f = bp_fx_u64tof32(select(a, bp_fx_neg64(a), n));
    return select(f, -f, n);
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
    ("add64", H_ADD64, &[]),
    ("neg64", H_NEG64, &[]),
    ("sub64", H_SUB64, &["add64", "neg64"]),
    ("abs64", H_ABS64, &["neg64"]),
    ("ult64", H_ULT64, &[]),
    ("slt64", H_SLT64, &[]),
    ("umin64", H_UMIN64, &["ult64"]),
    ("umax64", H_UMAX64, &["ult64"]),
    ("smin64", H_SMIN64, &["slt64"]),
    ("smax64", H_SMAX64, &["slt64"]),
    ("shl64", H_SHL64, &[]),
    ("lshr64", H_LSHR64, &[]),
    ("tdiv64", H_TDIV64, &["lshr64", "neg64"]),
    ("umulshr64", H_UMULSHR64, &["umul64"]),
    ("mul_u64", H_MUL_U64, &["umulshr64"]),
    ("mul_i64", H_MUL_I64, &["umulshr64", "neg64"]),
    ("udivshl64", H_UDIVSHL64, &["sub64", "ult64"]),
    ("div_u64", H_DIV_U64, &["udivshl64"]),
    ("div_i64", H_DIV_I64, &["udivshl64", "neg64"]),
    ("urem64", H_UREM64, &["sub64", "ult64"]),
    ("rem_i64", H_REM_I64, &["urem64", "neg64"]),
    ("i32to64", H_I32TO64, &[]),
    ("f2raw64", H_F2RAW64, &["neg64"]),
    ("u64tof32", H_U64TOF32, &[]),
    ("i64tof32", H_I64TOF32, &["u64tof32", "neg64"]),
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
        if f.width() > 64 {
            let msg = format!(
                "`{}` is not supported on --target wgpu: WGSL has no 64-bit integers; fixed-point formats \
                 are emulated up to 64 bits (e.g. `fixed16_16`, `fixed32_32`, `ufixed0_64`)",
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
            Some(r) if is64(f) => lit64(r),
            Some(r) => if f.signed { format!("i32({r})") } else { format!("{r}u") },
            None => {
                self.errors.push(TranspileError::at(
                    format!("literal out of range for {} (range [{}, {}])",
                        f.name(), f.format_exact(f.min_raw()), f.format_exact(f.max_raw())),
                    e.line, e.col));
                zero_const(f)
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
        if is64(f) { return self.fixed_binop64(op, &a, &b, f); }
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

    fn fixed_binop64(&mut self, op: &BinOp, a: &str, b: &str, f: &FixedFmt) -> String {
        let m = f.m;
        let sg = f.signed;
        let (lt, ltname) = if sg { ("bp_fx_slt64", "slt64") } else { ("bp_fx_ult64", "ult64") };
        let call = |this: &mut Self, name: &'static str, text: String| { this.use_fixed_helper(name); text };
        match op {
            BinOp::Add => call(self, "add64", format!("bp_fx_add64({a}, {b})")),
            BinOp::Sub => call(self, "sub64", format!("bp_fx_sub64({a}, {b})")),
            BinOp::Rem if sg => call(self, "rem_i64", format!("bp_fx_rem_i64({a}, {b})")),
            BinOp::Rem => call(self, "urem64", format!("bp_fx_urem64({a}, {b})")),
            BinOp::Mul if sg => call(self, "mul_i64", format!("bp_fx_mul_i64({a}, {b}, {m}u)")),
            BinOp::Mul => call(self, "mul_u64", format!("bp_fx_mul_u64({a}, {b}, {m}u)")),
            BinOp::Div if sg => call(self, "div_i64", format!("bp_fx_div_i64({a}, {b}, {m}u)")),
            BinOp::Div => call(self, "div_u64", format!("bp_fx_div_u64({a}, {b}, {m}u)")),
            BinOp::Eq => format!("all({a} == {b})"),
            BinOp::NotEq => format!("any({a} != {b})"),
            BinOp::Lt => call(self, ltname, format!("{lt}({a}, {b})")),
            BinOp::Gt => call(self, ltname, format!("{lt}({b}, {a})")),
            BinOp::LtEq => call(self, ltname, format!("!{lt}({b}, {a})")),
            BinOp::GtEq => call(self, ltname, format!("!{lt}({a}, {b})")),
            _ => unreachable!("fixed_binop_fmt filters the operators"),
        }
    }

    pub(super) fn fixed_neg(&mut self, operand: &Expr, f: &FixedFmt, e: &Expr) -> String {
        self.fixed_supported(f, e);
        let v = self.expr(operand);
        if is64(f) {
            self.use_fixed_helper("neg64");
            return format!("bp_fx_neg64({v})");
        }
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
                if is64(&f) { return Some(lit64(raw)); }
                return Some(if f.signed { format!("i32({raw})") } else { format!("{raw}u") });
            }
        }
        if field == "raw" {
            if let Some(f) = self.fixed_fmt_of(obj) {
                self.fixed_supported(&f, e);
                // `.raw` is an `int` (32-bit on this target); an unsigned 32-bit raw above 2^31 wraps.
                let o = self.expr(obj);
                if is64(&f) { return Some(format!("bitcast<i32>(({o}).x)")); }
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
        if is64(f) {
            return if f.signed {
                self.use_fixed_helper("tdiv64");
                format!("bp_fx_tdiv64({v}, {k}u)")
            } else {
                self.use_fixed_helper("lshr64");
                format!("bp_fx_lshr64({v}, {k}u)")
            };
        }
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
                        if is64(&f) {
                            let unsigned = self.ty_of(&a.value).as_ref().is_some_and(is_unsigned_int_ty);
                            return self.int_to_raw64(&n, unsigned);
                        }
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
        if is64(&f) { return self.fixed_method64(&f, &o, method, args, e); }
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

    /// An `int` expression sign-/zero-extended to a 64-bit raw pattern.
    fn int_to_raw64(&mut self, n: &str, unsigned: bool) -> String {
        if unsigned { format!("vec2<u32>(u32({n}), 0u)") }
        else { self.use_fixed_helper("i32to64"); format!("bp_fx_i32to64(i32({n}))") }
    }

    /// A raw of format `from` (any width) as a 64-bit limb pair, sign-/zero-extended.
    fn raw_to64(&mut self, from: &FixedFmt, v: &str) -> String {
        if is64(from) { v.to_string() }
        else if from.signed { self.use_fixed_helper("i32to64"); format!("bp_fx_i32to64({v})") }
        else { format!("vec2<u32>({v}, 0u)") }
    }

    /// The low bits of a 64-bit limb pair as `to`'s raw (re-narrowed below 32 bits).
    fn narrow64(&mut self, to: &FixedFmt, w: String) -> String {
        if is64(to) { return w; }
        let lo = if to.signed { format!("bitcast<i32>(({w}).x)") } else { format!("({w}).x") };
        wrap(to, lo)
    }

    fn fixed_method64(&mut self, f: &FixedFmt, o: &str, method: &str, args: &[Arg], e: &Expr) -> String {
        let m = f.m;
        let mask: u64 = if m >= 64 { u64::MAX } else { (1u64 << m) - 1 };
        let inv = lit64((!mask) as i128);
        let maskc = lit64(mask as i128);
        match (method, args) {
            ("floor", []) => format!("({o} & {inv})"),
            ("ceil", []) => { self.use_fixed_helper("add64"); format!("(bp_fx_add64({o}, {maskc}) & {inv})") }
            ("trunc", []) => {
                let q = self.fixed_trunc_div(f, o, m);
                self.use_fixed_helper("shl64");
                format!("bp_fx_shl64({q}, {m}u)")
            }
            ("fract", []) => {
                let q = self.fixed_trunc_div(f, o, m);
                self.use_fixed_helper("shl64");
                self.use_fixed_helper("sub64");
                format!("bp_fx_sub64({o}, bp_fx_shl64({q}, {m}u))")
            }
            ("abs", []) if f.signed => { self.use_fixed_helper("abs64"); format!("bp_fx_abs64({o})") }
            ("abs", []) => o.to_string(),
            ("min" | "max", [a]) => {
                let b = self.fixed_operand(&a.value, f);
                let name: &'static str = match (method, f.signed) {
                    ("min", true) => "smin64", ("max", true) => "smax64",
                    ("min", false) => "umin64", _ => "umax64",
                };
                self.use_fixed_helper(name);
                format!("bp_fx_{name}({o}, {b})")
            }
            _ => self.fixed_gpu_error(f, method, e),
        }
    }

    /// `x as T` between a 64-bit format and another fixed-point format.
    fn fixed_cast64(&mut self, from: &FixedFmt, to: &FixedFmt, v: &str) -> String {
        let w = self.raw_to64(from, v);
        let shifted = if to.m >= from.m {
            if to.m == from.m { w } else { self.use_fixed_helper("shl64"); format!("bp_fx_shl64({w}, {}u)", to.m - from.m) }
        } else {
            // A narrower signed source was sign-extended, so the 64-bit signed truncation is right.
            let ext = FixedFmt { signed: from.signed, n: 64 - from.m, m: from.m };
            self.fixed_trunc_div(&ext, &w, from.m - to.m)
        };
        self.narrow64(to, shifted)
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
                if is64(&from) || is64(&to) { return self.fixed_cast64(&from, &to, &v); }
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
            if is64(&to) {
                if is_float {
                    self.use_fixed_helper("f2raw64");
                    return format!("bp_fx_f2raw64(f32({v}) * {scale:.1})");
                }
                let unsigned = self.ty_of(inner).as_ref().is_some_and(is_unsigned_int_ty);
                let w = self.int_to_raw64(&v, unsigned);
                return if to.m == 0 { w } else {
                    self.use_fixed_helper("shl64");
                    format!("bp_fx_shl64({w}, {}u)", to.m)
                };
            }
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
            if is64(&from) {
                let h = if from.signed { "i64tof32" } else { "u64tof32" };
                self.use_fixed_helper(h);
                return format!("{dst}(bp_fx_{h}({v}) / {scale:.1})");
            }
            format!("{dst}(f32({v}) / {scale:.1})")
        } else if is64(&from) {
            let q = self.fixed_trunc_div(&from, &v, from.m);
            format!("{dst}(({q}).x)")
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
