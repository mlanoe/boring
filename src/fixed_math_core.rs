// Integer-only math for fixed-point values (sqrt, sin, cos, tan, exp, ln).
//
// docs/fixed-point-types.md, Phase 5. Bit-for-bit deterministic: shifts, adds and
// (at most) 120-bit products on `i128` — no floating point, no tables computed at
// run time. Internal precision is 60 fractional bits ("F"), so results are accurate
// to about 2^-54, then rounded to the target format's `m` fractional bits (to
// nearest, ties away from zero). `sqrt` is the exact floor.
//
// THIS FILE IS DUAL-USE: it is compiled as a module of `boring` (the Rust
// interpreter) AND pasted verbatim, inside `mod __fxm { ... }`, into every generated
// program that uses a fixed-point type. Keep it dependency-free (no `use`, no
// `crate::`, no inner attributes). The self-hosted interpreter re-implements the
// same algorithms in boring/interpreter/fixedpoint.br; a corpus case
// (tests/cases/fixed_point_math.br) keeps the three in agreement.

const F: u32 = 60;
const ONE: i128 = 1i128 << 60;
const HALF_PI: i128 = 1811004864519280711;
const LN2: i128 = 799144290325165979;
/// 1 / prod(sqrt(1 + 2^-2i)) for the 60 CORDIC iterations.
const CORDIC_K: i128 = 700114967507363238;
/// atan(2^-i) in F fractional bits.
const ATAN: [i128; 60] = [
    905502432259640355, 534549298976576474, 282441168888798124,
    143371547418228444, 71963988336308046, 36017075762092179,
    18012932708689205, 9007016009513623, 4503576721087964,
    2251796950380271, 1125899548928887, 562949908682076,
    281474971118251, 140737487656277, 70368744090283,
    35184372077909, 17592186043051, 8796093022037,
    4398046511083, 2199023255549, 1099511627776,
    549755813888, 274877906944, 137438953472,
    68719476736, 34359738368, 17179869184,
    8589934592, 4294967296, 2147483648,
    1073741824, 536870912, 268435456,
    134217728, 67108864, 33554432,
    16777216, 8388608, 4194304,
    2097152, 1048576, 524288,
    262144, 131072, 65536,
    32768, 16384, 8192,
    4096, 2048, 1024,
    512, 256, 128,
    64, 32, 16,
    8, 4, 2,
];

/// `raw` (m fractional bits) as F fractional bits.
fn up(raw: i128, m: u32) -> i128 {
    if m <= F { raw << (F - m) } else { raw >> (m - F) }
}

/// F-fractional-bit value as `m` fractional bits, rounded to nearest (ties away
/// from zero). `None` if it does not fit an `i128`.
fn down(v: i128, m: u32) -> Option<i128> {
    if m >= F {
        return v.checked_mul(1i128 << (m - F));
    }
    let s = F - m;
    let q = ((v.unsigned_abs() + (1u128 << (s - 1))) >> s) as i128;
    Some(if v < 0 { -q } else { q })
}

fn isqrt(mut n: u128) -> u128 {
    let mut x = 0u128;
    let mut bit = 1u128 << 126;
    while bit > n { bit >>= 2; }
    while bit != 0 {
        if n >= x + bit {
            n -= x + bit;
            x = (x >> 1) + bit;
        } else {
            x >>= 1;
        }
        bit >>= 2;
    }
    x
}

/// Exact floor of `sqrt(raw / 2^m) * 2^m`. `None` for a negative value.
pub fn __fxm_sqrt(raw: i128, m: u32) -> Option<i128> {
    if raw < 0 { return None; }
    Some(isqrt((raw as u128) << m) as i128)
}

/// (sin, cos) of `raw / 2^m` radians, in F fractional bits.
fn sincos(raw: i128, m: u32) -> (i128, i128) {
    let a = up(raw, m);
    // Nearest multiple of pi/2, then rotate the remainder (|r| <= pi/4) by CORDIC.
    let q = (a + HALF_PI / 2).div_euclid(HALF_PI);
    let r = a - q * HALF_PI;
    let (mut x, mut y, mut z) = (CORDIC_K, 0i128, r);
    let mut i = 0;
    while i < 60 {
        let (dx, dy) = (y >> i, x >> i);
        if z >= 0 {
            x -= dx;
            y += dy;
            z -= ATAN[i];
        } else {
            x += dx;
            y -= dy;
            z += ATAN[i];
        }
        i += 1;
    }
    match q.rem_euclid(4) {
        0 => (y, x),
        1 => (x, -y),
        2 => (-y, -x),
        _ => (-x, y),
    }
}

pub fn __fxm_sin(raw: i128, m: u32) -> Option<i128> {
    down(sincos(raw, m).0, m)
}

pub fn __fxm_cos(raw: i128, m: u32) -> Option<i128> {
    down(sincos(raw, m).1, m)
}

/// `None` where cos(x) rounds to zero, or on overflow.
pub fn __fxm_tan(raw: i128, m: u32) -> Option<i128> {
    let (s, c) = sincos(raw, m);
    if c == 0 { return None; }
    down((s << F) / c, m)
}

/// e^x. `None` on overflow; underflows to 0.
pub fn __fxm_exp(raw: i128, m: u32) -> Option<i128> {
    let x = up(raw, m);
    if x > (45i128 << F) { return None; }
    if x < -(43i128 << F) { return Some(0); }
    // x = k*ln2 + r with |r| <= ln2/2, then Taylor series for e^r.
    let k = (x + LN2 / 2).div_euclid(LN2);
    let r = x - k * LN2;
    let mut term = ONE;
    let mut sum = ONE;
    let mut n = 1i128;
    while n <= 24 {
        term = (term * r) >> F;
        term /= n;
        sum += term;
        n += 1;
    }
    let v = if k >= 0 { sum << (k as u32) } else { sum >> ((-k) as u32) };
    down(v, m)
}

/// Natural logarithm. `None` for a non-positive value.
pub fn __fxm_ln(raw: i128, m: u32) -> Option<i128> {
    let x = up(raw, m);
    if x <= 0 { return None; }
    // x = v * 2^e with v in [1, 2); ln(v) = 2 * atanh((v-1)/(v+1)).
    let mut e = 0i128;
    let mut v = x;
    while v >= (2i128 << F) { v >>= 1; e += 1; }
    while v < ONE { v <<= 1; e -= 1; }
    let z = ((v - ONE) << F) / (v + ONE);
    let z2 = (z * z) >> F;
    let mut term = z;
    let mut sum = z;
    let mut n = 3i128;
    while n <= 61 {
        term = (term * z2) >> F;
        sum += term / n;
        n += 2;
    }
    down(e * LN2 + 2 * sum, m)
}
