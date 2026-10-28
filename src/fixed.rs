//! Binary fixed-point scalar types `fixedN_M` / `ufixedN_M` (aliases `iN_M` /
//! `uN_M`) — see `docs/fixed-point-types.md`.
//!
//! This module is the single source of truth for the *shape* of a format (name
//! parsing, validity, range) and for the reference arithmetic. The interpreter
//! uses it directly; the transpiler emits Rust that must agree with it.
//!
//! A value of format `(signed, n, m)` is stored as a raw integer `r` of width
//! `w = n + m` bits; its real value is `r / 2^m`.

/// Widths a fixed-point format may occupy (`n + m`). 128 is accepted by the
/// checker but not implemented by any backend yet (docs, Phasing §4).
pub const VALID_WIDTHS: [u32; 5] = [8, 16, 32, 64, 128];

/// The integer math function named `name` (`sqrt`, `sin`, ...) from
/// `fixed_math_core.rs`, as `fn(raw, m) -> Option<raw at m fractional bits>`.
pub fn math_fn(name: &str) -> Option<fn(i128, u32) -> Option<i128>> {
    use crate::fixed_math_core as c;
    Some(match name {
        "sqrt" => c::__fxm_sqrt,
        "sin" => c::__fxm_sin,
        "cos" => c::__fxm_cos,
        "tan" => c::__fxm_tan,
        "exp" => c::__fxm_exp,
        "ln" => c::__fxm_ln,
        _ => return None,
    })
}

/// The four arithmetic operations the overflow-policy variants exist for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FxOp { Add, Sub, Mul, Div }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FixedFmt {
    pub signed: bool,
    /// Integer bits, including the sign bit for a signed format.
    pub n: u32,
    /// Fractional bits.
    pub m: u32,
}

/// Why a name that *looks* like a fixed-point type is not a valid one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixedNameError {
    /// `n + m` is not one of 8/16/32/64/128.
    BadWidth { n: u32, m: u32 },
    /// A signed format needs at least the sign bit.
    NoSignBit,
}

impl FixedNameError {
    pub fn message(&self, name: &str) -> String {
        match self {
            FixedNameError::BadWidth { n, m } => format!(
                "invalid fixed-point type `{name}`: integer bits + fractional bits = {} + {} = {}, \
                 which must be one of 8, 16, 32, 64 or 128",
                n, m, n + m
            ),
            FixedNameError::NoSignBit => format!(
                "invalid fixed-point type `{name}`: a signed fixed-point type needs at least 1 \
                 integer bit (the sign bit); use `ufixed0_M` for a pure fraction"
            ),
        }
    }
}

/// Splits `digits_digits` (`16_16`) into its two numbers. Returns `None` for
/// anything that is not exactly two plain digit runs joined by one `_`.
fn split_two_numbers(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.split_once('_')?;
    let plain = |p: &str| !p.is_empty() && p.len() <= 3 && p.bytes().all(|c| c.is_ascii_digit());
    if !plain(a) || !plain(b) {
        return None;
    }
    // `i08_8`-style zero padding is not a spelling of the type.
    if (a.len() > 1 && a.starts_with('0')) || (b.len() > 1 && b.starts_with('0')) {
        return None;
    }
    Some((a.parse().ok()?, b.parse().ok()?))
}

/// Recognises the *shape* of a fixed-point type name without judging validity:
/// `fixedN_M`, `ufixedN_M`, `iN_M`, `uN_M`.
fn shape(name: &str) -> Option<(bool, u32, u32)> {
    let (signed, rest) = if let Some(r) = name.strip_prefix("ufixed") {
        (false, r)
    } else if let Some(r) = name.strip_prefix("fixed") {
        (true, r)
    } else if let Some(r) = name.strip_prefix('i') {
        (true, r)
    } else if let Some(r) = name.strip_prefix('u') {
        (false, r)
    } else {
        return None;
    };
    let (n, m) = split_two_numbers(rest)?;
    Some((signed, n, m))
}

impl FixedFmt {
    /// Parses a type name. `None` means "not a fixed-point type name at all"
    /// (so it may be an ordinary identifier); `Some(Err(_))` means the name has
    /// the fixed-point shape but violates the validity rules — reserved names
    /// must be rejected, not silently treated as user types.
    pub fn parse(name: &str) -> Option<Result<FixedFmt, FixedNameError>> {
        let (signed, n, m) = shape(name)?;
        let w = n + m;
        if !VALID_WIDTHS.contains(&w) {
            return Some(Err(FixedNameError::BadWidth { n, m }));
        }
        if signed && n == 0 {
            return Some(Err(FixedNameError::NoSignBit));
        }
        Some(Ok(FixedFmt { signed, n, m }))
    }

    /// True when `name` must be treated as a type name by the parser: any
    /// `fixedN_M` / `ufixedN_M` shape (so an invalid one is *reported* rather than
    /// becoming an unknown identifier), but the short `iN_M` / `uN_M` spellings
    /// only when valid, so a variable that merely looks like `i3_4` is left alone.
    pub fn is_reserved_type_name(name: &str) -> bool {
        match Self::parse(name) {
            None => false,
            Some(Ok(_)) => true,
            Some(Err(_)) => name.starts_with("fixed") || name.starts_with("ufixed"),
        }
    }

    /// Like [`parse`](Self::parse) but only the valid case.
    pub fn from_name(name: &str) -> Option<FixedFmt> {
        Self::parse(name).and_then(Result::ok)
    }

    pub fn width(&self) -> u32 {
        self.n + self.m
    }

    /// Canonical spelling (`fixed16_16`, `ufixed8_8`).
    pub fn name(&self) -> String {
        format!("{}fixed{}_{}", if self.signed { "" } else { "u" }, self.n, self.m)
    }

    /// Smallest / largest raw storage value. 128-bit formats are not implemented
    /// anywhere; their bounds are clamped to the `i128` domain so queries never
    /// overflow (an unsigned 128-bit maximum is not representable in `i128`).
    pub fn min_raw(&self) -> i128 {
        if !self.signed { 0 } else if self.width() >= 128 { i128::MIN } else { -(1i128 << (self.width() - 1)) }
    }
    pub fn max_raw(&self) -> i128 {
        let w = self.width();
        if w >= 127 && (w >= 128 || self.signed) { return i128::MAX; }
        if self.signed { (1i128 << (w - 1)) - 1 } else { (1i128 << w) - 1 }
    }

    pub fn in_range(&self, raw: i128) -> bool {
        raw >= self.min_raw() && raw <= self.max_raw()
    }

    /// Rust storage type of the raw integer (`i32`, `u16`, …).
    pub fn rust_int(&self) -> String {
        format!("{}{}", if self.signed { "i" } else { "u" }, self.width())
    }

    /// Name of the generated Rust newtype (`Fixed16_16`, `UFixed8_8`).
    pub fn rust_name(&self) -> String {
        format!("{}Fixed{}_{}", if self.signed { "" } else { "U" }, self.n, self.m)
    }

    // ---- reference arithmetic (all results range-checked: `None` = overflow) ----

    pub fn checked(&self, raw: i128) -> Option<i128> {
        self.in_range(raw).then_some(raw)
    }

    pub fn add(&self, a: i128, b: i128) -> Option<i128> {
        self.checked(a.checked_add(b)?)
    }
    pub fn sub(&self, a: i128, b: i128) -> Option<i128> {
        self.checked(a.checked_sub(b)?)
    }
    pub fn neg(&self, a: i128) -> Option<i128> {
        self.checked(a.checked_neg()?)
    }

    /// `(ra * rb) >> m`, truncating toward zero. Done on magnitudes in `u128`,
    /// which holds the 2W-bit intermediate for every width up to 64.
    pub fn mul(&self, a: i128, b: i128) -> Option<i128> {
        if self.width() > 64 {
            return None;
        }
        let neg = (a < 0) != (b < 0);
        let prod = a.unsigned_abs().checked_mul(b.unsigned_abs())?;
        let mag = prod >> self.m;
        self.checked(Self::apply_sign(mag, neg)?)
    }

    /// `(ra << m) / rb`, truncating toward zero. `None` on overflow; the caller
    /// reports division by zero before calling.
    pub fn div(&self, a: i128, b: i128) -> Option<i128> {
        if self.width() > 64 || b == 0 {
            return None;
        }
        let neg = (a < 0) != (b < 0);
        let num = a.unsigned_abs().checked_shl(self.m)?;
        // `checked_shl` only checks the shift amount, not lost high bits.
        if num >> self.m != a.unsigned_abs() {
            return None;
        }
        let mag = num / b.unsigned_abs();
        self.checked(Self::apply_sign(mag, neg)?)
    }

    pub fn rem(&self, a: i128, b: i128) -> Option<i128> {
        if b == 0 {
            return None;
        }
        self.checked(a.checked_rem(b)?)
    }

    fn apply_sign(mag: u128, neg: bool) -> Option<i128> {
        if neg {
            if mag > (i128::MAX as u128) + 1 { None } else { Some((mag as i128).wrapping_neg()) }
        } else {
            i128::try_from(mag).ok()
        }
    }

    // ---- conversions ----

    /// Raw value of an exact integer, if it fits.
    pub fn from_int(&self, i: i128) -> Option<i128> {
        let raw = i.checked_mul(1i128.checked_shl(self.m)?)?;
        self.checked(raw)
    }

    /// Raw value of a decimal literal, **rounded to nearest, ties away from
    /// zero** (docs: Literals). `text` is the literal as written, with digit
    /// separators already removed. Exact: uses integer arithmetic on the
    /// decimal digits, so `0.1` never goes through an `f64`.
    pub fn from_decimal(&self, text: &str) -> Option<i128> {
        let (neg, body) = match text.strip_prefix('-') {
            Some(r) => (true, r),
            None => (false, text.strip_prefix('+').unwrap_or(text)),
        };
        let (int_part, frac_part) = body.split_once('.').unwrap_or((body, ""));
        if int_part.is_empty() && frac_part.is_empty() {
            return None;
        }
        let digits = format!("{int_part}{frac_part}");
        let num: u128 = digits.parse().ok()?;
        let den: u128 = 10u128.checked_pow(frac_part.len() as u32)?;
        // raw = round(num * 2^m / den). 2^m with m <= 128 overflows u128 only
        // at m = 128, which no implemented format uses.
        let scale = 1u128.checked_shl(self.m).filter(|_| self.m < 128)?;
        let scaled = num.checked_mul(scale)?;
        let (q, r) = (scaled / den, scaled % den);
        let q = if r >= den - r { q + 1 } else { q }; // ties round up in magnitude
        self.checked(Self::apply_sign(q, neg)?)
    }

    /// float → fixed, truncating toward zero (`as`). `None` for NaN/inf/range.
    pub fn from_f64_trunc(&self, f: f64) -> Option<i128> {
        if !f.is_finite() {
            return None;
        }
        let scaled = (f * 2f64.powi(self.m as i32)).trunc();
        if scaled < -(2f64.powi(126)) || scaled > 2f64.powi(126) {
            return None;
        }
        self.checked(scaled as i128)
    }

    pub fn to_f64(&self, raw: i128) -> f64 {
        raw as f64 / 2f64.powi(self.m as i32)
    }

    /// fixed → int, truncating toward zero.
    pub fn to_int(&self, raw: i128) -> i128 {
        let neg = raw < 0;
        let mag = (raw.unsigned_abs() >> self.m) as i128;
        if neg { -mag } else { mag }
    }

    /// fixed → other fixed: shift by the difference of `m`, truncating toward
    /// zero when fractional bits are dropped; `None` if the integer part does
    /// not fit the target.
    pub fn convert_to(&self, raw: i128, to: &FixedFmt) -> Option<i128> {
        let out = if to.m >= self.m {
            raw.checked_mul(1i128.checked_shl(to.m - self.m)?)?
        } else {
            let neg = raw < 0;
            let mag = (raw.unsigned_abs() >> (self.m - to.m)) as i128;
            if neg { -mag } else { mag }
        };
        to.checked(out)
    }

    /// Exact decimal expansion with trailing zeros trimmed (`1.5`,
    /// `0.0000152587890625`, `-3`). Always finite: the denominator is `2^m`.
    pub fn format_exact(&self, raw: i128) -> String {
        let neg = raw < 0;
        let mag = raw.unsigned_abs();
        let int_part = mag >> self.m;
        let mut frac = if self.m == 0 { 0 } else { mag & ((1u128 << self.m) - 1) };
        let mut s = String::new();
        if neg {
            s.push('-');
        }
        s.push_str(&int_part.to_string());
        if frac != 0 {
            s.push('.');
            // frac / 2^m: each ×10 step peels one exact decimal digit.
            while frac != 0 {
                frac *= 10;
                s.push(char::from(b'0' + (frac >> self.m) as u8));
                frac &= (1u128 << self.m) - 1;
            }
        }
        s
    }

    /// `{x:.Nf}` — rounded to nearest (ties away from zero) to `places`.
    pub fn format_places(&self, raw: i128, places: usize) -> String {
        let neg = raw < 0;
        let mag = raw.unsigned_abs();
        // value = mag / 2^m; scaled = round(mag * 10^places / 2^m)
        let pow10 = 10u128.pow(places as u32);
        let (hi, lo) = (mag >> self.m, if self.m == 0 { 0 } else { mag & ((1u128 << self.m) - 1) });
        // hi * 10^p is exact; lo * 10^p / 2^m needs rounding (lo < 2^64 here).
        let lo_scaled = lo * pow10;
        let half = if self.m == 0 { 0 } else { 1u128 << (self.m - 1) };
        let frac_digits = if self.m == 0 { 0 } else { (lo_scaled + half) >> self.m };
        let total = hi * pow10 + frac_digits;
        let s = total.to_string();
        let s = if s.len() <= places { format!("{}{s}", "0".repeat(places + 1 - s.len())) } else { s };
        let (ip, fp) = s.split_at(s.len() - places);
        let sign = if neg && total != 0 { "-" } else { "" };
        if places == 0 { format!("{sign}{ip}") } else { format!("{sign}{ip}.{fp}") }
    }
}

/// Rust source shared by every generated fixed-point newtype: the conversion
/// traits and the exact decimal formatter. Emitted once, when at least one
/// format is used.
pub const RUST_SHARED_PRELUDE: &str = r#"
// ---- fixed-point support (generated; see docs/fixed-point-types.md) ----
#[allow(dead_code)]
pub trait __FixedFormat: Sized + Copy {
    const FRAC_BITS: u32;
    fn __from_i128(v: i128) -> Option<Self>;
    fn __from_scaled(raw: i128, from_m: u32) -> Option<Self>;
    fn __from_f64(f: f64) -> Option<Self>;
    fn __raw128(self) -> i128;
    fn __to_f64(self) -> f64;
    fn __to_i128(self) -> i128;
}
#[allow(dead_code)]
pub trait __FixedCast: Sized {
    fn __to_fixed<F: __FixedFormat>(self) -> Option<F>;
}
macro_rules! __fixed_cast_int { ($($t:ty),*) => { $(
    impl __FixedCast for $t { fn __to_fixed<F: __FixedFormat>(self) -> Option<F> { F::__from_i128(self as i128) } }
)* } }
__fixed_cast_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, usize);
impl __FixedCast for u128 {
    fn __to_fixed<F: __FixedFormat>(self) -> Option<F> { i128::try_from(self).ok().and_then(F::__from_i128) }
}
impl __FixedCast for bool {
    fn __to_fixed<F: __FixedFormat>(self) -> Option<F> { F::__from_i128(self as i128) }
}
impl __FixedCast for f64 {
    fn __to_fixed<F: __FixedFormat>(self) -> Option<F> { F::__from_f64(self) }
}
impl __FixedCast for f32 {
    fn __to_fixed<F: __FixedFormat>(self) -> Option<F> { F::__from_f64(self as f64) }
}
#[allow(dead_code)]
fn __fx_exact(op: u8, a: i128, b: i128, m: u32, round: bool) -> (u128, bool) {
    match op {
        0 => { let v = a + b; (v.unsigned_abs(), v < 0) }
        1 => { let v = a - b; (v.unsigned_abs(), v < 0) }
        2 => {
            let prod = a.unsigned_abs() * b.unsigned_abs();
            let mut q = prod >> m;
            if round && m > 0 && (prod & ((1u128 << m) - 1)) >= (1u128 << (m - 1)) { q += 1; }
            (q, (a < 0) != (b < 0))
        }
        _ => {
            let den = b.unsigned_abs();
            let num = a.unsigned_abs() << m;
            let mut q = num / den;
            if round && (num % den) * 2 >= den { q += 1; }
            (q, (a < 0) != (b < 0))
        }
    }
}
#[allow(dead_code)]
fn __fx_chk(mag: u128, neg: bool, min: i128, max: i128) -> Option<i128> {
    let v = if neg {
        if mag > (i128::MAX as u128) + 1 { return None; }
        (mag as i128).wrapping_neg()
    } else {
        i128::try_from(mag).ok()?
    };
    if v < min || v > max { None } else { Some(v) }
}
#[allow(dead_code)]
fn __fx_sat(mag: u128, neg: bool, min: i128, max: i128) -> i128 {
    match __fx_chk(mag, neg, min, max) { Some(v) => v, None if neg => min, None => max }
}
#[allow(dead_code)]
fn __fx_wrap(mag: u128, neg: bool, w: u32, signed: bool) -> i128 {
    let mask: u128 = (1u128 << w) - 1;
    let mut low = mag & mask;
    if neg { low = low.wrapping_neg() & mask; }
    if signed && (low >> (w - 1)) == 1 { (low as i128).wrapping_sub(1i128 << w) } else { low as i128 }
}
/// Exact decimal expansion of `mag / 2^m` (trailing zeros trimmed), or rounded to
/// `{:.N}` places (ties away from zero) when a precision is given.
#[allow(dead_code)]
fn __fx_fmt(f: &mut core::fmt::Formatter<'_>, neg: bool, mag: u128, m: u32) -> core::fmt::Result {
    let mask: u128 = if m == 0 { 0 } else { (1u128 << m) - 1 };
    let mut int_part = mag >> m;
    let mut frac = mag & mask;
    let mut digits = [0u8; 96];
    let mut nd = 0usize;
    let mut round_up = false;
    match f.precision() {
        Some(p) => {
            let p = if p > 90 { 90 } else { p };
            while nd < p {
                frac *= 10;
                digits[nd] = (frac >> m) as u8;
                frac &= mask;
                nd += 1;
            }
            round_up = m != 0 && frac >= (1u128 << (m - 1));
        }
        None => {
            while frac != 0 {
                frac *= 10;
                digits[nd] = (frac >> m) as u8;
                frac &= mask;
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
    let is_zero = int_part == 0 && digits[..nd].iter().all(|d| *d == 0);
    let mut buf = [0u8; 136];
    let mut n = 0usize;
    let mut tmp = [0u8; 40];
    let mut t = 0usize;
    let mut v = int_part;
    loop {
        tmp[t] = b'0' + (v % 10) as u8;
        t += 1;
        v /= 10;
        if v == 0 { break; }
    }
    while t > 0 { t -= 1; buf[n] = tmp[t]; n += 1; }
    if nd > 0 {
        buf[n] = b'.';
        n += 1;
        for d in &digits[..nd] { buf[n] = b'0' + *d; n += 1; }
    }
    f.pad_integral(!(neg && !is_zero), "", core::str::from_utf8(&buf[..n]).unwrap())
}
"#;

impl FixedFmt {
    /// Rust type used for the 2W-bit intermediate of `*` and `/`.
    fn rust_wide(&self) -> &'static str {
        match (self.signed, self.width() <= 32) {
            (true, true) => "i64",
            (false, true) => "u64",
            (true, false) => "i128",
            (false, false) => "u128",
        }
    }

    /// Generated Rust newtype + operator impls for this format. Semantics must
    /// match the reference arithmetic above (truncate toward zero, debug panic /
    /// release wrap on overflow). Not available for 128-bit formats.
    pub fn rust_definition(&self) -> String {
        let (name, s, wd, m) = (self.rust_name(), self.rust_int(), self.rust_wide(), self.m);
        let neg = if self.signed {
            format!("impl core::ops::Neg for {name} {{ type Output = Self; fn neg(self) -> Self {{ Self(-self.0) }} }}\n")
        } else {
            String::new()
        };

        let round_body = if m == 0 {
            "self".to_string()
        } else {
            format!("{{ let r = self.0 as i128; let mag = r.unsigned_abs(); let mut ip = mag >> {m}; \
                      if (mag & ((1u128 << {m}) - 1)) >= (1u128 << ({m} - 1)) {{ ip += 1; }} \
                      let v = (ip << {m}) as i128; Self(Self::__narrow128(if r < 0 {{ -v }} else {{ v }})) }}")
        };
        let abs_body = if self.signed { "Self(self.0.abs())" } else { "self" };
        let (w, sg, minr, maxr) = (self.width(), self.signed, self.min_raw(), self.max_raw());
        let mut policy = format!(
            "    fn __ex(self, op: u8, o: Self, round: bool) -> (u128, bool) {{ __fx_exact(op, self.0 as i128, o.0 as i128, {m}, round) }}\n\
                 fn __round_result(m: u128, n: bool) -> Self {{\n\
                     match __fx_chk(m, n, {minr}i128, {maxr}i128) {{\n\
                         Some(v) => Self(v as {s}),\n\
                         None => if cfg!(debug_assertions) {{ panic!(\"fixed-point overflow\") }} else {{ Self(__fx_wrap(m, n, {w}, {sg}) as {s}) }},\n\
                     }}\n\
                 }}\n\
                 pub fn mul_round(self, o: Self) -> Self {{ let (m, n) = self.__ex(2, o, true); Self::__round_result(m, n) }}\n\
                 pub fn div_round(self, o: Self) -> Self {{ let (m, n) = self.__ex(3, o, true); Self::__round_result(m, n) }}\n");
        policy.push_str(&format!(
            "    fn __fxm_fin(v: Option<i128>, what: &str) -> Self {{\n\
                     match v {{\n\
                         Some(v) => if cfg!(debug_assertions) {{ Self(<{s}>::try_from(v).expect(\"fixed-point overflow\")) }} else {{ Self(v as {s}) }},\n\
                         None => panic!(\"{{}}\", what),\n\
                     }}\n\
                 }}\n"));
        for (name, msg) in [
            ("sqrt", "square root of a negative value"), ("sin", "fixed-point overflow"),
            ("cos", "fixed-point overflow"), ("tan", "tan: cos is zero here"),
            ("exp", "fixed-point overflow"), ("ln", "logarithm of a non-positive value"),
        ] {
            policy.push_str(&format!(
                "    pub fn {name}(self) -> Self {{ Self::__fxm_fin(__fxm::__fxm_{name}(self.0 as i128, {m}), \"{msg}\") }}\n"));
        }
        for (opn, code) in [("add", 0), ("sub", 1), ("mul", 2), ("div", 3)] {
            policy.push_str(&format!(
                "    pub fn wrapping_{opn}(self, o: Self) -> Self {{ let (m, n) = self.__ex({code}, o, false); Self(__fx_wrap(m, n, {w}, {sg}) as {s}) }}\n\
                     pub fn saturating_{opn}(self, o: Self) -> Self {{ let (m, n) = self.__ex({code}, o, false); Self(__fx_sat(m, n, {minr}i128, {maxr}i128) as {s}) }}\n\
                     pub fn checked_{opn}(self, o: Self) -> Option<Self> {{ {zero_guard}let (m, n) = self.__ex({code}, o, false); __fx_chk(m, n, {minr}i128, {maxr}i128).map(|v| Self(v as {s})) }}\n",
                zero_guard = if code == 3 { "if o.0 == 0 { return None; } " } else { "" }));
        }
        let mut out = format!(
r#"
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct {name}(pub {s});
#[allow(dead_code)]
impl {name} {{
    pub const FRAC_BITS: u32 = {m};
    pub const EPSILON: Self = Self(1);
    pub const MIN: Self = Self({s}::MIN);
    pub const MAX: Self = Self({s}::MAX);
    pub const fn from_raw(r: {s}) -> Self {{ Self(r) }}
    pub const fn raw(self) -> {s} {{ self.0 }}
    fn __narrow(w: {wd}) -> {s} {{
        if cfg!(debug_assertions) {{ <{s}>::try_from(w).expect("fixed-point overflow") }} else {{ w as {s} }}
    }}
    fn __narrow128(w: i128) -> {s} {{
        if cfg!(debug_assertions) {{ <{s}>::try_from(w).expect("fixed-point overflow") }} else {{ w as {s} }}
    }}
    pub fn trunc(self) -> Self {{ let one = 1i128 << {m}; Self((((self.0 as i128) / one) * one) as {s}) }}
    pub fn floor(self) -> Self {{
        let one = 1i128 << {m}; let r = self.0 as i128; let t = (r / one) * one;
        Self(Self::__narrow128(if r < t {{ t - one }} else {{ t }}))
    }}
    pub fn ceil(self) -> Self {{
        let one = 1i128 << {m}; let r = self.0 as i128; let t = (r / one) * one;
        Self(Self::__narrow128(if r > t {{ t + one }} else {{ t }}))
    }}
    pub fn round(self) -> Self {{ {round_body} }}
    pub fn fract(self) -> Self {{ Self(self.0 - self.trunc().0) }}
    pub fn abs(self) -> Self {{ {abs_body} }}
    pub fn signum(self) -> Self {{
        <Self as __FixedFormat>::__from_i128((self.0 as i128).signum()).expect("fixed-point overflow")
    }}
    pub fn min(self, o: Self) -> Self {{ core::cmp::Ord::min(self, o) }}
    pub fn max(self, o: Self) -> Self {{ core::cmp::Ord::max(self, o) }}
    pub fn clamp(self, lo: Self, hi: Self) -> Self {{ core::cmp::Ord::clamp(self, lo, hi) }}
{policy}}}
impl core::ops::Add for {name} {{ type Output = Self; fn add(self, o: Self) -> Self {{ Self(self.0 + o.0) }} }}
impl core::ops::Sub for {name} {{ type Output = Self; fn sub(self, o: Self) -> Self {{ Self(self.0 - o.0) }} }}
impl core::ops::Mul for {name} {{
    type Output = Self;
    fn mul(self, o: Self) -> Self {{ Self(Self::__narrow(((self.0 as {wd}) * (o.0 as {wd})) / ((1 as {wd}) << {m}))) }}
}}
impl core::ops::Div for {name} {{
    type Output = Self;
    fn div(self, o: Self) -> Self {{ Self(Self::__narrow(((self.0 as {wd}) << {m}) / (o.0 as {wd}))) }}
}}
impl core::ops::Rem for {name} {{ type Output = Self; fn rem(self, o: Self) -> Self {{ Self(self.0 % o.0) }} }}
impl core::ops::AddAssign for {name} {{ fn add_assign(&mut self, o: Self) {{ *self = *self + o; }} }}
impl core::ops::SubAssign for {name} {{ fn sub_assign(&mut self, o: Self) {{ *self = *self - o; }} }}
impl core::ops::MulAssign for {name} {{ fn mul_assign(&mut self, o: Self) {{ *self = *self * o; }} }}
impl core::ops::DivAssign for {name} {{ fn div_assign(&mut self, o: Self) {{ *self = *self / o; }} }}
impl core::ops::RemAssign for {name} {{ fn rem_assign(&mut self, o: Self) {{ *self = *self % o; }} }}
impl core::fmt::Display for {name} {{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {{
        let v = self.0 as i128;
        __fx_fmt(f, v < 0, v.unsigned_abs(), {m})
    }}
}}
impl core::fmt::Debug for {name} {{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {{ core::fmt::Display::fmt(self, f) }}
}}
impl __FixedFormat for {name} {{
    const FRAC_BITS: u32 = {m};
    fn __from_i128(v: i128) -> Option<Self> {{
        v.checked_mul(1i128 << {m}).and_then(|r| <{s}>::try_from(r).ok()).map(Self)
    }}
    fn __from_scaled(raw: i128, from_m: u32) -> Option<Self> {{
        let r = if {m} >= from_m {{ raw.checked_mul(1i128 << ({m} - from_m))? }} else {{ raw / (1i128 << (from_m - {m})) }};
        <{s}>::try_from(r).ok().map(Self)
    }}
    fn __from_f64(f: f64) -> Option<Self> {{
        if !f.is_finite() {{ return None; }}
        let scaled = f * ((1u128 << {m}) as f64);
        <{s}>::try_from(scaled as i128).ok().map(Self)
    }}
    fn __raw128(self) -> i128 {{ self.0 as i128 }}
    fn __to_f64(self) -> f64 {{ (self.0 as f64) / ((1u128 << {m}) as f64) }}
    fn __to_i128(self) -> i128 {{ (self.0 as i128) / (1i128 << {m}) }}
}}
impl __FixedCast for {name} {{
    fn __to_fixed<F: __FixedFormat>(self) -> Option<F> {{ F::__from_scaled(self.0 as i128, {m}) }}
}}
"#);
        out.push_str(&neg);
        out
    }

    // ---- rounding / sign helpers (docs: Members). `None` = not representable ----

    fn one_raw(&self) -> Option<i128> {
        1i128.checked_shl(self.m)
    }

    /// Toward zero.
    pub fn trunc(&self, raw: i128) -> Option<i128> {
        self.checked(self.to_int(raw).checked_mul(self.one_raw()?)?)
    }

    pub fn floor(&self, raw: i128) -> Option<i128> {
        let t = self.trunc(raw)?;
        if raw < t { self.checked(t.checked_sub(self.one_raw()?)?) } else { Some(t) }
    }

    pub fn ceil(&self, raw: i128) -> Option<i128> {
        let t = self.trunc(raw)?;
        if raw > t { self.checked(t.checked_add(self.one_raw()?)?) } else { Some(t) }
    }

    /// Nearest integer, ties away from zero.
    pub fn round(&self, raw: i128) -> Option<i128> {
        if self.m == 0 {
            return Some(raw);
        }
        let neg = raw < 0;
        let mag = raw.unsigned_abs();
        let mut ip = mag >> self.m;
        if mag & ((1u128 << self.m) - 1) >= (1u128 << (self.m - 1)) {
            ip += 1;
        }
        self.checked(Self::apply_sign(ip.checked_shl(self.m)?, neg)?)
    }

    /// `x - trunc(x)`: carries the sign of `x`.
    pub fn fract(&self, raw: i128) -> Option<i128> {
        self.checked(raw - self.trunc(raw)?)
    }

    pub fn abs(&self, raw: i128) -> Option<i128> {
        self.checked(raw.checked_abs()?)
    }

    /// -1, 0 or 1 as a fixed value (`None` when 1 is not representable, e.g. `ufixed0_16`).
    pub fn signum(&self, raw: i128) -> Option<i128> {
        match raw.cmp(&0) {
            std::cmp::Ordering::Less => self.from_int(-1),
            std::cmp::Ordering::Equal => Some(0),
            std::cmp::Ordering::Greater => self.from_int(1),
        }
    }

    // ---- overflow policies (docs: Phase 2): wrapping / saturating / checked, and
    // round-to-nearest `*` and `/`. All start from the exact result of the
    // operation as (magnitude, negative), before any range check. ----

    /// Exact `a op b` (truncated toward zero, or rounded to nearest with ties away
    /// from zero when `round`) as magnitude + sign. `None` for width > 64 or a
    /// zero divisor.
    pub fn exact(&self, op: FxOp, a: i128, b: i128, round: bool) -> Option<(u128, bool)> {
        if self.width() > 64 {
            return None;
        }
        Some(match op {
            FxOp::Add => ((a + b).unsigned_abs(), a + b < 0),
            FxOp::Sub => ((a - b).unsigned_abs(), a - b < 0),
            FxOp::Mul => {
                let prod = a.unsigned_abs() * b.unsigned_abs();
                let mut q = prod >> self.m;
                if round && self.m > 0 && prod & ((1u128 << self.m) - 1) >= (1u128 << (self.m - 1)) {
                    q += 1;
                }
                (q, (a < 0) != (b < 0))
            }
            FxOp::Div => {
                if b == 0 {
                    return None;
                }
                let den = b.unsigned_abs();
                let num = a.unsigned_abs() << self.m;
                let mut q = num / den;
                if round && (num % den) * 2 >= den {
                    q += 1;
                }
                (q, (a < 0) != (b < 0))
            }
        })
    }

    /// In-range result, or `None` (the `checked*` policy).
    pub fn checked_from(&self, mag: u128, neg: bool) -> Option<i128> {
        self.checked(Self::apply_sign(mag, neg)?)
    }

    /// Two's-complement truncation to the storage width (the `wrapping*` policy).
    pub fn wrapping_from(&self, mag: u128, neg: bool) -> i128 {
        let w = self.width();
        let mask = if w >= 128 { u128::MAX } else { (1u128 << w) - 1 };
        let mut low = mag & mask;
        if neg {
            low = low.wrapping_neg() & mask;
        }
        if self.signed && low >> (w - 1) == 1 {
            (low as i128).wrapping_sub(1i128 << w)
        } else {
            low as i128
        }
    }

    /// Clamped to the representable range (the `saturating*` policy).
    pub fn saturating_from(&self, mag: u128, neg: bool) -> i128 {
        match self.checked_from(mag, neg) {
            Some(v) => v,
            None if neg => self.min_raw(),
            None => self.max_raw(),
        }
    }

    /// An untyped `f64` literal as the author wrote it: Rust's `{:?}` is the
    /// shortest decimal that round-trips (`0.1` -> "0.1"), so rounding that text
    /// exactly implements "literals round to nearest" without any f64 error.
    pub fn from_f64_literal(&self, f: f64) -> Option<i128> {
        if !f.is_finite() {
            return None;
        }
        let text = format!("{f:?}");
        let text = if text.contains(['e', 'E']) { format!("{f:.20}") } else { text };
        self.from_decimal(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(name: &str) -> FixedFmt {
        FixedFmt::from_name(name).unwrap()
    }

    #[test]
    fn names_and_aliases() {
        assert_eq!(f("fixed16_16"), f("i16_16"));
        assert_eq!(f("ufixed8_8"), f("u8_8"));
        assert_eq!(f("ufixed0_16").width(), 16);
        assert_eq!(f("fixed16_16").name(), "fixed16_16");
        assert_eq!(f("u8_8").name(), "ufixed8_8");
    }

    #[test]
    fn not_a_fixed_name() {
        for n in ["int32", "i32", "u8", "fixed", "fixed16", "fixed_16", "iabc_8", "i16_", "Fixed16_16", "i016_16", "u8_8_8"] {
            assert!(FixedFmt::parse(n).is_none(), "{n}");
        }
    }

    #[test]
    fn invalid_but_reserved() {
        assert!(matches!(FixedFmt::parse("fixed10_10"), Some(Err(FixedNameError::BadWidth { .. }))));
        assert!(matches!(FixedFmt::parse("fixed0_16"), Some(Err(FixedNameError::NoSignBit))));
        assert!(FixedFmt::parse("ufixed0_16").unwrap().is_ok());
    }

    #[test]
    fn range() {
        let t = f("fixed8_8");
        assert_eq!((t.min_raw(), t.max_raw()), (-32768, 32767));
        let u = f("ufixed8_8");
        assert_eq!((u.min_raw(), u.max_raw()), (0, 65535));
    }

    #[test]
    fn literal_rounding() {
        let t = f("fixed16_16");
        assert_eq!(t.from_decimal("1.5"), Some(98304));
        assert_eq!(t.from_decimal("-1.5"), Some(-98304));
        // 0.1 * 65536 = 6553.6 -> 6554
        assert_eq!(t.from_decimal("0.1"), Some(6554));
        assert_eq!(t.from_decimal("-0.1"), Some(-6554));
        // ties away from zero: 2^-17 = half an lsb
        assert_eq!(t.from_decimal("0.000007629394531250"), Some(1));
        assert_eq!(t.from_decimal("32768.0"), None);
        assert_eq!(f("fixed8_8").from_decimal("200.5"), None);
        assert_eq!(f("ufixed8_8").from_decimal("200.5"), Some(51328));
    }

    #[test]
    fn mul_div_truncate_toward_zero() {
        let t = f("fixed16_16");
        let a = t.from_decimal("1.5").unwrap();
        let b = t.from_decimal("2.25").unwrap();
        assert_eq!(t.format_exact(t.mul(a, b).unwrap()), "3.375");
        assert_eq!(t.mul(-a, b), t.mul(a, b).map(|x| -x)); // symmetric
        assert_eq!(t.format_exact(t.div(a, b).unwrap()), "0.666656494140625");
        assert_eq!(t.mul(t.max_raw(), t.max_raw()), None); // overflow
    }

    #[test]
    fn wide_unsigned_mul() {
        let u = f("ufixed32_32");
        let a = u.from_int(3).unwrap();
        let b = u.from_decimal("1.5").unwrap();
        assert_eq!(u.format_exact(u.mul(a, b).unwrap()), "4.5");
        assert_eq!(u.mul(u.max_raw(), u.from_int(1).unwrap()), Some(u.max_raw()));
    }

    #[test]
    fn formatting() {
        let t = f("fixed16_16");
        assert_eq!(t.format_exact(98304), "1.5");
        assert_eq!(t.format_exact(1), "0.0000152587890625");
        assert_eq!(t.format_exact(-98304), "-1.5");
        assert_eq!(t.format_exact(3 << 16), "3");
        assert_eq!(t.format_places(t.from_decimal("1.0625").unwrap(), 3), "1.063");
        assert_eq!(t.format_places(t.from_decimal("-1.5").unwrap(), 0), "-2");
        assert_eq!(t.format_places(t.from_decimal("0.001").unwrap(), 1), "0.0");
    }

    #[test]
    fn rounding_helpers() {
        let t = f("fixed16_16");
        let r = |s: &str| t.from_decimal(s).unwrap();
        let show = |v: Option<i128>| t.format_exact(v.unwrap());
        assert_eq!(show(t.floor(r("-1.25"))), "-2");
        assert_eq!(show(t.floor(r("1.75"))), "1");
        assert_eq!(show(t.ceil(r("-1.25"))), "-1");
        assert_eq!(show(t.ceil(r("1.25"))), "2");
        assert_eq!(show(t.trunc(r("-1.75"))), "-1");
        assert_eq!(show(t.round(r("2.5"))), "3");
        assert_eq!(show(t.round(r("-2.5"))), "-3");
        assert_eq!(show(t.round(r("2.4"))), "2");
        assert_eq!(show(t.fract(r("-1.25"))), "-0.25");
        assert_eq!(show(t.abs(r("-1.25"))), "1.25");
        assert_eq!(show(t.signum(r("-9"))), "-1");
        assert_eq!(t.ceil(t.max_raw()), None); // 32767.99998 -> 32768 overflows
        assert_eq!(f("ufixed0_16").signum(5), None);
    }

    #[test]
    fn overflow_policies() {
        let t = f("fixed8_8"); // range [-128, 127.99609375]
        let r = |s: &str| t.from_decimal(s).unwrap();
        let show = |v: i128| t.format_exact(v);
        let (a, b) = (r("100"), r("2"));
        let (m, n) = t.exact(FxOp::Mul, a, b, false).unwrap();
        assert_eq!(t.checked_from(m, n), None);
        assert_eq!(show(t.saturating_from(m, n)), "127.99609375");
        assert_eq!(show(t.wrapping_from(m, n)), "-56"); // 200 - 256
        let (m, n) = t.exact(FxOp::Mul, a, -b, false).unwrap();
        assert_eq!(show(t.saturating_from(m, n)), "-128");
        let (m, n) = t.exact(FxOp::Add, r("1.5"), r("2.25"), false).unwrap();
        assert_eq!(show(t.checked_from(m, n).unwrap()), "3.75");
        assert_eq!(t.exact(FxOp::Div, 1, 0, false), None);
        // rounding: 1 lsb * 0.5 = half an lsb -> truncates to 0, rounds to 1
        let half = r("0.5");
        let (m, n) = t.exact(FxOp::Mul, 1, half, false).unwrap();
        assert_eq!(t.checked_from(m, n), Some(0));
        let (m, n) = t.exact(FxOp::Mul, 1, half, true).unwrap();
        assert_eq!(t.checked_from(m, n), Some(1));
        let (m, n) = t.exact(FxOp::Div, r("1"), r("3"), true).unwrap();
        assert_eq!(t.checked_from(m, n), Some(85)); // 0.33203125 (truncation gives 85 too); 2/3 below
        let (m, n) = t.exact(FxOp::Div, r("2"), r("3"), true).unwrap();
        assert_eq!(t.checked_from(m, n), Some(171)); // 170.67 rounds up
        let (m, n) = t.exact(FxOp::Div, r("2"), r("3"), false).unwrap();
        assert_eq!(t.checked_from(m, n), Some(170));
        let u = f("ufixed8_8");
        let (m, n) = u.exact(FxOp::Sub, 1, 2, false).unwrap();
        assert_eq!(u.wrapping_from(m, n), 65535);
        assert_eq!(u.saturating_from(m, n), 0);
    }

    /// Compares the integer math against f64 at `m` fractional bits (<= 2 ulp).
    fn close(got: Option<i128>, want: f64, m: u32, what: &str) {
        let got = got.unwrap_or_else(|| panic!("{what}: None"));
        let want_raw = (want * (1u128 << m) as f64).round() as i128;
        assert!((got - want_raw).abs() <= 2, "{what}: got {got}, want {want_raw}");
    }

    #[test]
    fn integer_math_matches_f64() {
        use crate::fixed_math_core::*;
        let m = 32u32;
        let raw = |x: f64| (x * (1u128 << m) as f64).round() as i128;
        // The reference is evaluated at the quantized input, like the integer code sees it.
        let q = |x: f64| raw(x) as f64 / (1u128 << m) as f64;
        for x in [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.14159, 10.0, 100.5, 1234.5678] {
            close(__fxm_sqrt(raw(x), m), q(x).sqrt(), m, "sqrt");
        }
        for x in [0.0, 0.3, 1.0, 1.5707963, 2.0, 3.0, 3.14159, 6.0, 7.5, -0.5, -2.0, -100.0, 300.0] {
            close(__fxm_sin(raw(x), m), q(x).sin(), m, &format!("sin {x}"));
            close(__fxm_cos(raw(x), m), q(x).cos(), m, &format!("cos {x}"));
        }
        for x in [0.0, 0.5, 1.0, -1.0, 1.4, -1.4, 3.0] {
            close(__fxm_tan(raw(x), m), q(x).tan(), m, &format!("tan {x}"));
        }
        for x in [0.0, 0.5, 1.0, -1.0, 2.5, -3.7, 10.0, -20.0] {
            close(__fxm_exp(raw(x), m), q(x).exp(), m, &format!("exp {x}"));
        }
        for x in [0.001, 0.5, 1.0, 2.0, 2.718281828, 10.0, 1234.5, 1.0e6] {
            close(__fxm_ln(raw(x), m), q(x).ln(), m, &format!("ln {x}"));
        }
        assert_eq!(__fxm_sqrt(-1, m), None);
        assert_eq!(__fxm_ln(0, m), None);
        assert_eq!(__fxm_exp(raw(50.0), m), None);
    }

    #[test]
    fn conversions() {
        let a = f("fixed16_16");
        let b = f("fixed8_24");
        let x = a.from_decimal("1.75").unwrap();
        assert_eq!(b.format_exact(a.convert_to(x, &b).unwrap()), "1.75");
        assert_eq!(a.to_int(a.from_decimal("-1.75").unwrap()), -1);
        assert_eq!(a.from_f64_trunc(-1.99999), Some(-131071));
        assert_eq!(a.from_f64_trunc(f64::NAN), None);
        assert_eq!(f("fixed8_8").convert_to(32767, &f("fixed4_12")), None);
    }
}
