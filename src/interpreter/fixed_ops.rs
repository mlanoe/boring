//! Members of fixed-point values (`x.raw`, `x.floor()`, `fixed16_16.fromRaw(n)`,
//! `fixed16_16.epsilon`, ...) — docs/fixed-point-types.md, Members.

use super::*;
use crate::fixed::{FixedFmt, FxOp};

/// The raw storage integer as the interpreter value of the same width and sign.
pub(crate) fn raw_as_value(raw: i128, fmt: &FixedFmt) -> Value {
    match (fmt.signed, fmt.width()) {
        (true, 8) => Value::Int8(raw as i8),
        (true, 16) => Value::Int16(raw as i16),
        (true, 32) => Value::Int32(raw as i32),
        (true, 64) => Value::Int64(raw as i64),
        (false, 8) => Value::Uint8(raw as u8),
        (false, 16) => Value::Uint16(raw as u16),
        (false, 32) => Value::Uint32(raw as u32),
        (false, 64) => Value::Uint64(raw as u64),
        _ => Value::Int128(raw),
    }
}

fn value_as_raw_int(v: &Value) -> Option<i128> {
    Some(match v {
        Value::Int(n) => *n as i128,
        Value::Uint(n) => *n as i128,
        Value::Uint8(n) => *n as i128,
        Value::Int8(n) => *n as i128,
        Value::Int16(n) => *n as i128,
        Value::Int32(n) => *n as i128,
        Value::Int64(n) => *n as i128,
        Value::Int128(n) => *n,
        Value::Uint16(n) => *n as i128,
        Value::Uint32(n) => *n as i128,
        Value::Uint64(n) => *n as i128,
        Value::Uint128(n) => i128::try_from(*n).ok()?,
        _ => return None,
    })
}

impl Interpreter {
    /// `fixed16_16.epsilon` / `.min` / `.max`.
    pub(crate) fn fixed_constant(fmt: &FixedFmt, name: &str) -> Option<Value> {
        let raw = match name {
            "epsilon" => 1,
            "min" => fmt.min_raw(),
            "max" => fmt.max_raw(),
            _ => return None,
        };
        Some(Value::Fixed(raw, *fmt))
    }

    /// `fixed16_16.fromRaw(n)`: reinterprets an integer as raw storage (must fit).
    pub(crate) fn fixed_static_call(fmt: &FixedFmt, method: &str, args: &[Value], line: usize) -> Eval {
        match (method, args) {
            ("fromRaw", [a]) => {
                let raw = value_as_raw_int(a)
                    .ok_or_else(|| err(format!("{}.fromRaw: expected an integer, got {}", fmt.name(), a.type_name()), line))?;
                if !fmt.in_range(raw) {
                    return Err(err(format!("{}.fromRaw: {} does not fit the {}-bit storage", fmt.name(), raw, fmt.width()), line));
                }
                Ok(Value::Fixed(raw, *fmt))
            }
            _ => Err(err(format!("{} has no function '{}'", fmt.name(), method), line)),
        }
    }

    /// Instance methods; `None` when `method` is not one (falls through to the
    /// generic path and its "unknown method" diagnostics).
    pub(crate) fn fixed_method(raw: i128, fmt: &FixedFmt, method: &str, args: &[Value], line: usize) -> Option<Eval> {
        let wrap = |r: Option<i128>, what: &str| -> Eval {
            r.map(|r| Value::Fixed(r, *fmt))
                .ok_or_else(|| err(format!("fixed-point overflow in {}.{}()", fmt.name(), what), line))
        };
        // A bare literal argument takes the receiver's format (as in operators).
        let operand = |v: &Value| -> Result<i128, Signal> {
            match v {
                Value::Fixed(r, f) if f == fmt => Ok(*r),
                Value::Fixed(_, f) => Err(err(format!("cannot mix {} and {}; cast one with `as`", fmt.name(), f.name()), line)),
                Value::Int(n) => fmt.from_int(*n as i128).ok_or_else(|| err(format!("literal out of range for {}", fmt.name()), line)),
                Value::Float64(x) => fmt.from_f64_literal(*x).ok_or_else(|| err(format!("literal out of range for {}", fmt.name()), line)),
                other => Err(err(format!("{}.{}: expected {}, got {}", fmt.name(), method, fmt.name(), other.type_name()), line)),
            }
        };
        // wrappingAdd / saturatingMul / checkedDiv / mulRound / divRound ...
        let policy_op = |m: &str| -> Option<(&str, FxOp, bool)> {
            let (policy, rest) = ["wrapping", "saturating", "checked"].iter()
                .find_map(|p| m.strip_prefix(p).map(|r| (*p, r)))?;
            let op = match rest { "Add" => FxOp::Add, "Sub" => FxOp::Sub, "Mul" => FxOp::Mul, "Div" => FxOp::Div, _ => return None };
            Some((policy, op, false))
        };
        if let ([a], Some((policy, op, _))) = (args, policy_op(method)) {
            return Some((|| {
                let b = operand(a)?;
                if op == FxOp::Div && b == 0 {
                    return if policy == "checked" { Ok(Value::Nil) } else { Err(err("division by zero", line)) };
                }
                let (mag, neg) = fmt.exact(op, raw, b, false)
                    .ok_or_else(|| err(format!("{} `{}` is not supported for 128-bit fixed-point formats", fmt.name(), method), line))?;
                Ok(match policy {
                    "wrapping" => Value::Fixed(fmt.wrapping_from(mag, neg), *fmt),
                    "saturating" => Value::Fixed(fmt.saturating_from(mag, neg), *fmt),
                    _ => fmt.checked_from(mag, neg).map(|r| Value::Fixed(r, *fmt)).unwrap_or(Value::Nil),
                })
            })());
        }
        if let ([a], "mulRound" | "divRound") = (args, method) {
            return Some((|| {
                let b = operand(a)?;
                let op = if method == "mulRound" { FxOp::Mul } else { FxOp::Div };
                if op == FxOp::Div && b == 0 {
                    return Err(err("division by zero", line));
                }
                let (mag, neg) = fmt.exact(op, raw, b, true)
                    .ok_or_else(|| err(format!("{} `{}` is not supported for 128-bit fixed-point formats", fmt.name(), method), line))?;
                fmt.checked_from(mag, neg).map(|r| Value::Fixed(r, *fmt))
                    .ok_or_else(|| err(format!("fixed-point overflow in {}.{}()", fmt.name(), method), line))
            })());
        }
        // sqrt / sin / cos / tan / exp / ln (integer algorithms: src/fixed_math_core.rs)
        if let (Some(f), []) = (crate::fixed::math_fn(method), args) {
            return Some((|| {
                let m = fmt.m;
                match f(raw, m) {
                    Some(v) => fmt.checked(v).map(|r| Value::Fixed(r, *fmt))
                        .ok_or_else(|| err(format!("fixed-point overflow in {}.{}()", fmt.name(), method), line)),
                    None => Err(err(match method {
                        "sqrt" => format!("{}.sqrt(): square root of a negative value", fmt.name()),
                        "ln" => format!("{}.ln(): logarithm of a non-positive value", fmt.name()),
                        "tan" => format!("{}.tan(): cos is zero here", fmt.name()),
                        _ => format!("fixed-point overflow in {}.{}()", fmt.name(), method),
                    }, line)),
                }
            })());
        }
        Some(match (method, args) {
            ("floor", []) => wrap(fmt.floor(raw), "floor"),
            ("ceil", []) => wrap(fmt.ceil(raw), "ceil"),
            ("round", []) => wrap(fmt.round(raw), "round"),
            ("trunc", []) => wrap(fmt.trunc(raw), "trunc"),
            ("fract", []) => wrap(fmt.fract(raw), "fract"),
            ("abs", []) => wrap(fmt.abs(raw), "abs"),
            ("signum", []) => wrap(fmt.signum(raw), "signum"),
            ("min", [a]) => operand(a).map(|b| Value::Fixed(raw.min(b), *fmt)),
            ("max", [a]) => operand(a).map(|b| Value::Fixed(raw.max(b), *fmt)),
            ("clamp", [lo, hi]) => (|| {
                let (lo, hi) = (operand(lo)?, operand(hi)?);
                if lo > hi {
                    return Err(err(format!("{}.clamp: min is greater than max", fmt.name()), line));
                }
                Ok(Value::Fixed(raw.clamp(lo, hi), *fmt))
            })(),
            _ => return None,
        })
    }
}
