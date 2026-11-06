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

// Fixed-point (`fixedN_M` / `ufixedN_M`) support for the Linux (`--linux`) backend.
//
// The kernel emitter has no type checker, so this keeps a flat name -> format map
// (params, `let`s, struct fields, function return types) and resolves the format of an
// expression from it. The generated code uses the same integer-only newtypes and math
// core as the std backend (`fixed_emit::fixed_point_prelude`), minus every `f32`/`f64`
// item (`fixed_point_prelude_kernel`): the target has no FPU. A float is never
// converted at runtime; a decimal literal written next to a fixed-point operand
// (`x * 1.5`) is not accepted by the validator, and `<float literal> as fixedN_M` is
// folded to its raw integer at transpile time.

use std::collections::HashSet;

use crate::ast::{Arg, BinOp, Expr, ExprKind, Type, UnaryOp};
use crate::fixed::FixedFmt;
use super::helpers::KernelTranspiler;

pub(super) fn fmt_of_type(ty: &Type) -> Option<FixedFmt> {
    match ty {
        Type::Fixed(f) => Some(*f),
        Type::Named(n) => FixedFmt::from_name(n),
        Type::Mut(inner) | Type::Qualified(inner, _) => fmt_of_type(inner),
        _ => None,
    }
}

impl KernelTranspiler {
    /// Records (or forgets, when `ty`/`init` is not fixed-point) the format of a binding.
    pub(super) fn track_fixed(&mut self, name: &str, ty: Option<&Type>, init: Option<&Expr>) {
        let fmt = ty.and_then(fmt_of_type).or_else(|| init.and_then(|e| self.fixed_fmt_of(e)));
        match fmt {
            Some(f) => { self.fixed_vars.insert(name.to_string(), f); }
            None => { self.fixed_vars.remove(name); }
        }
    }

    pub(super) fn fixed_fmt_of(&self, e: &Expr) -> Option<FixedFmt> {
        match &e.kind {
            ExprKind::Var(v) => self.fixed_vars.get(v).copied(),
            // `fixed16_16.epsilon` / `.min` / `.max` are constants of that format.
            ExprKind::Field(obj, f) if matches!(f.as_str(), "epsilon" | "min" | "max")
                && matches!(&obj.kind, ExprKind::Var(v) if FixedFmt::from_name(v).is_some()) =>
                match &obj.kind { ExprKind::Var(v) => FixedFmt::from_name(v), _ => None },
            ExprKind::Field(_, f) => self.fixed_fields.get(f).copied(),
            ExprKind::Cast(_, ty) => fmt_of_type(ty),
            ExprKind::Call(callee, _) => match &callee.kind {
                ExprKind::Var(n) => self.fixed_fns.get(n).copied(),
                _ => None,
            },
            ExprKind::BinOp(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem, l, r) =>
                self.fixed_fmt_of(l).or_else(|| self.fixed_fmt_of(r)),
            ExprKind::UnaryOp(UnaryOp::Neg, inner) => self.fixed_fmt_of(inner),
            ExprKind::MethodCall(obj, m, _) => {
                let builtin = matches!(m.as_str(),
                    "floor" | "ceil" | "round" | "trunc" | "fract" | "abs" | "signum" | "min" | "max" | "clamp"
                    | "sqrt" | "sin" | "cos" | "tan" | "exp" | "ln" | "mulRound" | "divRound")
                    || m.starts_with("wrapping") || m.starts_with("saturating");
                // A fixed-point receiver means the member; otherwise a user method's declared return.
                if builtin { if let Some(f) = self.fixed_fmt_of(obj) { return Some(f); } }
                self.fixed_methods.get(m).copied()
            }
            _ => None,
        }
    }

    /// `Some(raw)` for an untyped integer literal (optionally negated); the inner option is
    /// "fits `fmt`". Float literals only fold when `allow_float` (the `<lit> as fixed` cast).
    fn fixed_literal_raw(e: &Expr, fmt: &FixedFmt, allow_float: bool) -> Option<Option<i128>> {
        match &e.kind {
            ExprKind::Int(n) => Some(fmt.from_int(*n as i128)),
            ExprKind::Float(f) if allow_float => Some(fmt.from_f64_literal(*f)),
            ExprKind::UnaryOp(UnaryOp::Neg, inner) => match &inner.kind {
                ExprKind::Int(n) => Some(fmt.from_int(-(*n as i128))),
                ExprKind::Float(f) if allow_float => Some(fmt.from_f64_literal(-*f)),
                _ => None,
            },
            _ => None,
        }
    }

    pub(super) fn emit_fixed_literal(&self, e: &Expr, fmt: &FixedFmt, allow_float: bool) -> Option<String> {
        Some(match Self::fixed_literal_raw(e, fmt, allow_float)? {
            Some(raw) => format!("{}::from_raw({})", fmt.rust_name(), raw),
            None => {
                crate::transpiler::helpers::report_unsupported(e.line, e.col, format!(
                    "literal out of range for {} (range [{}, {}])",
                    fmt.name(), fmt.format_exact(fmt.min_raw()), fmt.format_exact(fmt.max_raw())));
                format!("{}::default()", fmt.rust_name())
            }
        })
    }

    /// Operand of a fixed-point operation: a bare int literal takes the operation's format.
    pub(super) fn emit_fixed_operand(&self, e: &Expr, fmt: &FixedFmt) -> String {
        self.emit_fixed_literal(e, fmt, false).unwrap_or_else(|| self.emit_expr(e))
    }

    /// `let fixed16_16 y = 3` against a declared fixed type.
    pub(super) fn emit_fixed_let_value(&self, ty: &Type, val: &Expr) -> Option<String> {
        let fmt = fmt_of_type(ty)?;
        self.emit_fixed_literal(val, &fmt, false)
    }

    pub(super) fn try_emit_fixed_binop(&self, op: &BinOp, l: &Expr, r: &Expr) -> Option<String> {
        let sym = match op {
            BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*", BinOp::Div => "/", BinOp::Rem => "%",
            BinOp::Eq => "==", BinOp::NotEq => "!=", BinOp::Lt => "<", BinOp::Gt => ">",
            BinOp::LtEq => "<=", BinOp::GtEq => ">=",
            _ => return None,
        };
        if matches!(l.kind, ExprKind::Nil) || matches!(r.kind, ExprKind::Nil) {
            return None;
        }
        let fmt = self.fixed_fmt_of(l).or_else(|| self.fixed_fmt_of(r))?;
        Some(format!("({} {} {})", self.emit_fixed_operand(l, &fmt), sym, self.emit_fixed_operand(r, &fmt)))
    }

    pub(super) fn try_emit_fixed_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        if let ExprKind::Var(v) = &obj.kind {
            if let Some(fmt) = FixedFmt::from_name(v) {
                return match (method, args) {
                    ("fromRaw", [a]) => Some(format!("{}::from_raw(({}) as {})",
                        fmt.rust_name(), self.emit_expr(&a.value), fmt.rust_int())),
                    _ => None,
                };
            }
        }
        let rust_method = match method {
            "floor" | "ceil" | "round" | "trunc" | "fract" | "abs" | "signum" | "min" | "max" | "clamp"
            | "sqrt" | "sin" | "cos" | "tan" | "exp" | "ln" => method.to_string(),
            "mulRound" => "mul_round".to_string(),
            "divRound" => "div_round".to_string(),
            m => {
                let (policy, op) = ["wrapping", "saturating", "checked"].iter()
                    .find_map(|p| m.strip_prefix(p).map(|r| (*p, r)))?;
                if !matches!(op, "Add" | "Sub" | "Mul" | "Div") {
                    return None;
                }
                format!("{}_{}", policy, op.to_lowercase())
            }
        };
        let fmt = self.fixed_fmt_of(obj)?;
        if fmt.width() >= 128 && matches!(method, "sqrt" | "sin" | "cos" | "tan" | "exp" | "ln") {
            crate::transpiler::helpers::report_unsupported(obj.line, obj.col, format!(
                "`.{method}()` is not supported for the 128-bit format {} (the integer math core is limited to 120-bit products) — use a format of at most 64 bits",
                fmt.name()));
        }
        let args_s: Vec<String> = args.iter().map(|a| self.emit_fixed_operand(&a.value, &fmt)).collect();
        Some(format!("{}.{}({})", self.emit_expr(obj), rust_method, args_s.join(", ")))
    }

    /// `<fixed>.checkedOp(y) else default`.
    pub(super) fn try_emit_fixed_else(&self, e: &Expr, default: &Expr) -> Option<String> {
        let ExprKind::MethodCall(obj, method, args) = &e.kind else { return None };
        if !method.starts_with("checked") {
            return None;
        }
        let fmt = self.fixed_fmt_of(obj)?;
        let call = self.try_emit_fixed_method(obj, method, args)?;
        Some(format!("{}.unwrap_or({})", call, self.emit_fixed_operand(default, &fmt)))
    }

    pub(super) fn try_emit_fixed_field(&self, obj: &Expr, field: &str) -> Option<String> {
        if let ExprKind::Var(v) = &obj.kind {
            if let Some(fmt) = FixedFmt::from_name(v) {
                let c = match field { "epsilon" => "EPSILON", "min" => "MIN", "max" => "MAX", _ => return None };
                return Some(format!("{}::{}", fmt.rust_name(), c));
            }
        }
        if field == "raw" && self.fixed_fmt_of(obj).is_some() {
            return Some(format!("{}.raw()", self.emit_expr(obj)));
        }
        None
    }

    /// `e as T` when `e` or `T` is fixed-point. A float-typed source or target is rejected
    /// (the validator normally stops these first).
    pub(super) fn try_emit_fixed_cast(&self, e: &Expr, ty: &Type) -> Option<String> {
        let src_fmt = self.fixed_fmt_of(e);
        if let Some(to) = fmt_of_type(ty) {
            if let Some(lit) = self.emit_fixed_literal(e, &to, true) {
                return Some(lit);
            }
            // Saturates (by the source's sign) instead of panicking: see `kernelize`.
            return Some(format!("__fx_cast_sat::<{}, _>({})", to.rust_name(), self.emit_expr(e)));
        }
        src_fmt?;
        let dst = self.emit_type(ty);
        if dst.contains("f32") || dst.contains("f64") {
            crate::transpiler::helpers::report_unsupported(e.line, e.col,
                "fixed-point to float conversion is not allowed in kernel context — FPU is disabled");
            return Some("0".into());
        }
        if matches!(ty.without_mut(), t if super::fixed::is_int_type(t)) {
            return Some(format!("(__FixedFormat::__to_i128({}) as {})", self.emit_expr(e), dst));
        }
        None
    }
}

pub(super) fn is_int_type(t: &Type) -> bool {
    matches!(t, Type::Int | Type::Uint | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
        | Type::Uint8 | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128)
        || matches!(t, Type::Named(n) if matches!(n.as_str(),
            "int" | "uint" | "isize" | "usize" | "int8" | "int16" | "int32" | "int64" | "int128"
            | "uint8" | "uint16" | "uint32" | "uint64" | "uint128"))
}

/// Integer-only fixed-point prelude for the Linux target: the shared helpers, the math core and
/// one newtype per format used in `generated`, with every `f32`/`f64` item removed.
pub(super) fn fixed_point_prelude_kernel(generated: &str) -> Option<String> {
    let (prelude, _) = crate::transpiler::fixed_emit::fixed_point_prelude(generated);
    let prelude = prelude?;
    Some(kernelize(&strip_float_items(&prelude)))
}

/// Makes the (float-free) prelude panic-free: a kernel module must never `BUG()`. The std
/// backend panics in debug builds and wraps in release builds; here every overflow wraps,
/// division and remainder by zero give 0, and a math function outside its domain saturates
/// (`sqrt` of a negative -> 0, `ln` of a non-positive -> `MIN`, overflow / `tan` at a pole -> `MAX`).
fn kernelize(src: &str) -> String {
    let mut s = replace_debug_branches(src);
    // 128-bit math stubs: a call is a transpile error (see `try_emit_fixed_method`), so the
    // stub body is never reached; it must still not panic.
    for f in ["sqrt", "sin", "cos", "tan", "exp", "ln"] {
        s = s.replace(&format!("{{ panic!(\"{f}() is not supported for 128-bit fixed-point formats\") }}"), "{ self }");
    }
    for (from, to) in [
        ("Self(self.0 + o.0)", "Self(self.0.wrapping_add(o.0))"),
        ("Self(self.0 - o.0)", "Self(self.0.wrapping_sub(o.0))"),
        ("Self(-self.0)", "Self(self.0.wrapping_neg())"),
        ("Self(self.0 % o.0)", "Self(if o.0 == 0 { 0 } else { self.0.wrapping_rem(o.0) })"),
        ("fn div(self, o: Self) -> Self { Self(Self::__narrow(", "fn div(self, o: Self) -> Self { if o.0 == 0 { return Self(0); } Self(Self::__narrow("),
        ("Self(self.0.abs())", "Self(self.0.wrapping_abs())"),
        ("None => panic!(\"{}\", what),",
         "None => if what.starts_with(\"logarithm\") { Self::MIN } else if what.starts_with(\"square\") { Self(0) } else { Self::MAX },"),
        ("<Self as __FixedFormat>::__from_i128((self.0 as i128).signum()).expect(\"fixed-point overflow\")",
         "{ let s = (self.0 as i128).signum(); <Self as __FixedFormat>::__from_i128(s).unwrap_or(if s < 0 { Self::MIN } else { Self::MAX }) }"),
        ("    fn __raw128(self) -> i128;\n", "    fn __raw128(self) -> i128;\n    fn __min() -> Self;\n    fn __max() -> Self;\n"),
        ("    fn __raw128(self) -> i128 { self.0 as i128 }\n", "    fn __raw128(self) -> i128 { self.0 as i128 }\n    fn __min() -> Self { Self::MIN }\n    fn __max() -> Self { Self::MAX }\n"),
        ("assert!(den != 0, \"division by zero\");", "if den == 0 { return (0, 0, 0); }"),
        ("(hi + 1, 0)", "(hi.wrapping_add(1), 0)"),
        ("(hi, lo + 1)", "(hi, lo.wrapping_add(1))"),
        ("core::str::from_utf8(&buf[..n]).unwrap()", "core::str::from_utf8(&buf[..n]).unwrap_or(\"\")"),
    ] {
        s = s.replace(from, to);
    }
    s.push_str(CAST_SAT);
    // `__FxSign` for every generated format (signed ones can be negative).
    let mut extra = String::new();
    for l in s.lines() {
        if let Some(name) = l.strip_prefix("impl __FixedCast for ").and_then(|r| r.strip_suffix(" {")) {
            if name.contains("Fixed") {
                let neg = if name.starts_with('U') { "false" } else { "self.0 < 0" };
                extra.push_str(&format!("impl __FxSign for {name} {{ fn __neg(&self) -> bool {{ {neg} }} }}\n"));
            }
        }
    }
    s.push_str(&extra);
    s
}

/// Saturating `T as fixedN_M` (the source's sign picks `MIN` / `MAX` when out of range).
const CAST_SAT: &str = r#"
#[allow(dead_code)]
trait __FxSign { fn __neg(&self) -> bool; }
macro_rules! __fx_sign_int { ($($t:ty),*) => { $(
    #[allow(unused_comparisons)]
    impl __FxSign for $t { fn __neg(&self) -> bool { *self < 0 } }
)* } }
__fx_sign_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);
impl __FxSign for bool { fn __neg(&self) -> bool { false } }
#[allow(dead_code)]
fn __fx_cast_sat<F: __FixedFormat, T: __FixedCast + __FxSign>(v: T) -> F {
    let neg = v.__neg();
    match v.__to_fixed::<F>() { Some(f) => f, None => if neg { F::__min() } else { F::__max() } }
}
"#;

/// `if cfg!(debug_assertions) { A } else { B }` -> `B` (the wrapping branch).
fn replace_debug_branches(src: &str) -> String {
    const HEAD: &str = "if cfg!(debug_assertions) {";
    let mut out = String::new();
    let mut rest = src;
    while let Some(at) = rest.find(HEAD) {
        out.push_str(&rest[..at]);
        let after = &rest[at + HEAD.len() - 1..]; // starts at the `{` of A
        let close_a = matching_brace(after);
        let tail = after[close_a + 1..].trim_start();
        let Some(tail) = tail.strip_prefix("else") else { out.push_str(HEAD); rest = &rest[at + HEAD.len()..]; continue };
        let tail = tail.trim_start();
        let close_b = matching_brace(tail);
        out.push_str(tail[1..close_b].trim());
        rest = &tail[close_b + 1..];
    }
    out.push_str(rest);
    out
}

/// Index of the `}` matching the `{` at the start of `s`.
fn matching_brace(s: &str) -> usize {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c { '{' => depth += 1, '}' => { depth -= 1; if depth == 0 { return i; } } _ => {} }
    }
    s.len() - 1
}

/// Removes the float entry points from the std prelude (see `fixed::RUST_SHARED_PRELUDE` and
/// `FixedFmt::rust_definition`): the `__from_f64`/`__to_f64` trait items and impls, and the
/// `impl __FixedCast for f64/f32` blocks.
fn strip_float_items(src: &str) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        let t = l.trim_start();
        // one-line trait items / one-line impls
        if t.starts_with("fn __from_f64(f: f64) -> Option<Self>;") || t.starts_with("fn __to_f64(self) -> f64;")
            || (t.starts_with("fn __to_f64(self)") && t.ends_with('}'))
        {
            i += 1;
            continue;
        }
        // multi-line `fn __from_f64(...) -> Option<Self> { ... }` (body at deeper indent, closes at `    }`)
        if t.starts_with("fn __from_f64(f: f64) -> Option<Self> {") {
            while !lines[i].starts_with("    }") { i += 1; }
            i += 1;
            continue;
        }
        if t.starts_with("impl __FixedCast for f64 {") || t.starts_with("impl __FixedCast for f32 {") {
            while !lines[i].starts_with('}') { i += 1; }
            i += 1;
            continue;
        }
        out.push_str(l);
        out.push('\n');
        i += 1;
    }
    out
}

/// `(name, format)` per parameter / field, in declaration order.
pub(super) type ParamFmts = Vec<(String, Option<FixedFmt>)>;

fn put<T: PartialEq>(m: &mut std::collections::HashMap<String, T>, conflict: &mut HashSet<String>, k: &str, v: T) {
    match m.get(k) {
        Some(old) if *old != v => { conflict.insert(k.to_string()); }
        _ => { m.insert(k.to_string(), v); }
    }
}

impl KernelTranspiler {
    /// Collects, program-wide, the fixed-point formats of struct fields, function and method
    /// return types and parameters. A name declared twice with different formats (overloads,
    /// two structs sharing a field name) is dropped: inference then falls back to "unknown".
    pub(super) fn collect_fixed_decls(&mut self, program: &crate::ast::Program) {
        use crate::ast::{FnDecl, Item};
        let mut c_fields = HashSet::new();
        let mut c_fns = HashSet::new();
        let mut c_methods = HashSet::new();
        let mut c_fp = HashSet::new();
        let mut c_mp = HashSet::new();
        let mut c_sf = HashSet::new();
        let params_of = |f: &FnDecl| -> ParamFmts {
            f.params.iter().filter(|p| p.name != "self")
                .map(|p| (p.name.clone(), p.ty.as_ref().and_then(fmt_of_type))).collect()
        };
        let mut methods: Vec<&FnDecl> = Vec::new();
        for item in &program.items {
            match item {
                Item::Fn(f) => {
                    if let Some(fm) = f.return_ty.as_ref().and_then(fmt_of_type) { put(&mut self.fixed_fns, &mut c_fns, &f.name, fm); }
                    put(&mut self.fixed_fn_params, &mut c_fp, &f.name, params_of(f));
                }
                Item::Struct(s) => {
                    for fd in &s.fields {
                        if let Some(fm) = fmt_of_type(&fd.ty) { put(&mut self.fixed_fields, &mut c_fields, &fd.name, fm); }
                    }
                    let fields: ParamFmts = s.fields.iter().map(|fd| (fd.name.clone(), fmt_of_type(&fd.ty))).collect();
                    put(&mut self.fixed_struct_fields, &mut c_sf, &s.name, fields);
                    methods.extend(s.methods.iter());
                }
                Item::Enum(e) => methods.extend(e.methods.iter()),
                Item::Ext(x) => methods.extend(x.methods.iter()),
                _ => {}
            }
        }
        for m in methods {
            if let Some(fm) = m.return_ty.as_ref().and_then(fmt_of_type) { put(&mut self.fixed_methods, &mut c_methods, &m.name, fm); }
            put(&mut self.fixed_method_params, &mut c_mp, &m.name, params_of(m));
        }
        for c in c_fields { self.fixed_fields.remove(&c); }
        for c in c_fns { self.fixed_fns.remove(&c); }
        for c in c_methods { self.fixed_methods.remove(&c); }
        for c in c_fp { self.fixed_fn_params.remove(&c); }
        for c in c_mp { self.fixed_method_params.remove(&c); }
        for c in c_sf { self.fixed_struct_fields.remove(&c); }
    }

    /// Arguments of a call whose parameter formats are `fmts`: a bare int literal in a
    /// fixed-point slot (by label, else by position) is folded to that format.
    pub(super) fn emit_args_with(&self, args: &[Arg], fmts: Option<&ParamFmts>) -> Vec<String> {
        args.iter().enumerate().map(|(i, a)| {
            let fmt = fmts.and_then(|f| match &a.label {
                Some(l) => f.iter().find(|(n, _)| n == l).and_then(|(_, fm)| *fm),
                None => f.get(i).and_then(|(_, fm)| *fm),
            });
            match fmt { Some(fm) => self.emit_fixed_operand(&a.value, &fm), None => self.emit_expr(&a.value) }
        }).collect()
    }

    /// A value in return position: folds a literal to the function's fixed-point return format.
    pub(super) fn emit_return_value(&self, v: &Expr) -> String {
        match &self.cur_ret_fixed { Some(fm) => self.emit_fixed_operand(v, fm), None => self.emit_expr(v) }
    }
}
