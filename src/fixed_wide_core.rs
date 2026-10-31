// 256-bit intermediates for fixed-point arithmetic (128-bit formats, docs Phase 4).
//
// Everything here works on *sign-magnitude* pairs: a magnitude in `u128` plus a sign
// flag, and — because a 128-bit result can need up to 256 bits before it is range
// checked — an `ovf` flag meaning "the true magnitude does not fit 128 bits; `mag`
// holds its low 128 bits". That is exactly what the overflow policies need:
//   checked    -> None if ovf (or out of range),
//   saturating -> min/max if ovf,
//   wrapping   -> the low bits (ovf is irrelevant).
//
// The same helpers serve every width (the narrow formats never overflow `u128`, so
// `ovf` is just false there), which keeps one code path for all formats.
//
// THIS FILE IS DUAL-USE, like fixed_math_core.rs: compiled as a module of `boring`
// (the Rust interpreter) AND pasted verbatim, inside `mod __fxw { ... }`, into every
// generated program that uses a fixed-point type. Keep it dependency-free (no `use`,
// no `crate::`, no inner attributes). The self-hosted interpreter ports it line for
// line in boring/interpreter/fixedpoint.br (`fxw_*`); tests/cases/fixed_point_wide.br
// keeps the three in agreement.
//
// Fractional bit counts passed in here are at most 126 (`m <= 126`); the format
// checker enforces it.

const M64: u128 = 0xFFFF_FFFF_FFFF_FFFF;

/// Full 256-bit product of two 128-bit values as `(hi, lo)`.
pub fn __fxw_mul_full(a: u128, b: u128) -> (u128, u128) {
    let (a0, a1, b0, b1) = (a & M64, a >> 64, b & M64, b >> 64);
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 64) + (p01 & M64) + (p10 & M64);
    let lo = (p00 & M64) | ((mid & M64) << 64);
    let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    (hi, lo)
}

/// `(hi, lo) >> s` for `s < 128`.
fn __fxw_shr(hi: u128, lo: u128, s: u32) -> (u128, u128) {
    if s == 0 { (hi, lo) } else { (hi >> s, (lo >> s) | (hi << (128 - s))) }
}

/// `x << s` as 256 bits, for `s < 128`.
fn __fxw_shl(x: u128, s: u32) -> (u128, u128) {
    if s == 0 { (0, x) } else { (x >> (128 - s), x << s) }
}

/// `(hi, lo) + 1`.
fn __fxw_inc(hi: u128, lo: u128) -> (u128, u128) {
    if lo == u128::MAX { (hi + 1, 0) } else { (hi, lo + 1) }
}

/// 256-by-128-bit long division: `(quotient hi, quotient lo, remainder)`.
pub fn __fxw_divrem(hi: u128, lo: u128, den: u128) -> (u128, u128, u128) {
    assert!(den != 0, "division by zero");
    if hi == 0 {
        return (0, lo / den, lo % den);
    }
    let (mut qh, mut ql, mut r) = (0u128, 0u128, 0u128);
    let mut i = 256u32;
    while i > 0 {
        i -= 1;
        let bit = if i >= 128 { (hi >> (i - 128)) & 1 } else { (lo >> i) & 1 };
        let carry = r >> 127;
        r = (r << 1) | bit;
        qh = (qh << 1) | (ql >> 127);
        ql <<= 1;
        if carry == 1 || r >= den {
            r = r.wrapping_sub(den);
            ql |= 1;
        }
    }
    (qh, ql, r)
}

/// Sign-magnitude `a + b`.
fn __fxw_addsm(ma: u128, na: bool, mb: u128, nb: bool) -> (u128, bool, bool) {
    if na == nb {
        let (s, o) = ma.overflowing_add(mb);
        (s, na, o)
    } else if ma >= mb {
        (ma - mb, na && ma != mb, false)
    } else {
        (mb - ma, nb, false)
    }
}

/// Exact `a + b` as `(magnitude, negative, overflowed 128 bits)`.
pub fn __fxw_add(a: i128, b: i128) -> (u128, bool, bool) {
    __fxw_addsm(a.unsigned_abs(), a < 0, b.unsigned_abs(), b < 0)
}

/// Exact `a - b`.
pub fn __fxw_sub(a: i128, b: i128) -> (u128, bool, bool) {
    __fxw_addsm(a.unsigned_abs(), a < 0, b.unsigned_abs(), b >= 0)
}

/// `(a * b) >> m`, truncated toward zero, or rounded to nearest (ties away from
/// zero) when `round`.
pub fn __fxw_mul(a: i128, b: i128, m: u32, round: bool) -> (u128, bool, bool) {
    let (hi, lo) = __fxw_mul_full(a.unsigned_abs(), b.unsigned_abs());
    let (mut qh, mut ql) = __fxw_shr(hi, lo, m);
    if round && m > 0 && ((lo >> (m - 1)) & 1) == 1 {
        let (h, l) = __fxw_inc(qh, ql);
        qh = h;
        ql = l;
    }
    (ql, (a < 0) != (b < 0), qh != 0)
}

/// `(a << m) / b`, truncated toward zero or rounded to nearest (ties away from
/// zero). `b` must be non-zero.
pub fn __fxw_div(a: i128, b: i128, m: u32, round: bool) -> (u128, bool, bool) {
    let den = b.unsigned_abs();
    let (nh, nl) = __fxw_shl(a.unsigned_abs(), m);
    let (mut qh, mut ql, r) = __fxw_divrem(nh, nl, den);
    if round && r >= den - r {
        let (h, l) = __fxw_inc(qh, ql);
        qh = h;
        ql = l;
    }
    (ql, (a < 0) != (b < 0), qh != 0)
}

/// `round(num * 2^m / den)` (ties up), the magnitude of a decimal literal; `None` if
/// it does not fit 128 bits.
pub fn __fxw_scale_decimal(num: u128, den: u128, m: u32) -> Option<u128> {
    let (nh, nl) = __fxw_shl(num, m);
    let (mut qh, mut ql, r) = __fxw_divrem(nh, nl, den);
    if r >= den - r {
        let (h, l) = __fxw_inc(qh, ql);
        qh = h;
        ql = l;
    }
    if qh != 0 { None } else { Some(ql) }
}

/// Decimal expansion of `mag / 2^m` as `(integer part, fraction digits, digit
/// count)`. Without `places` the expansion is exact with trailing zeros trimmed (at
/// most `m` digits, <= 126); with `places` it is cut to that many digits (at most
/// 128 — everything past `m` digits is zero anyway) and rounded to nearest, ties
/// away from zero, carrying into the integer part.
pub fn __fxw_digits(mag: u128, m: u32, places: Option<usize>) -> (u128, [u8; 130], usize) {
    let mask: u128 = (1u128 << m) - 1;
    let mut int_part = mag >> m;
    let mut frac = mag & mask;
    let mut digits = [0u8; 130];
    let mut nd = 0usize;
    let mut round_up = false;
    match places {
        Some(p) => {
            let p = if p > 128 { 128 } else { p };
            while nd < p {
                let (h, l) = __fxw_mul_full(frac, 10);
                digits[nd] = (if m == 0 { 0 } else { (h << (128 - m)) | (l >> m) }) as u8;
                frac = l & mask;
                nd += 1;
            }
            round_up = m != 0 && frac >= (1u128 << (m - 1));
        }
        None => {
            while frac != 0 {
                let (h, l) = __fxw_mul_full(frac, 10);
                digits[nd] = ((h << (128 - m)) | (l >> m)) as u8;
                frac = l & mask;
                nd += 1;
            }
        }
    }
    if round_up {
        let mut i = nd;
        loop {
            if i == 0 { int_part += 1; break; }
            i -= 1;
            if digits[i] == 9 { digits[i] = 0; } else { digits[i] += 1; break; }
        }
    }
    (int_part, digits, nd)
}
