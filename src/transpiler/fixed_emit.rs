//! Transpilation of binary fixed-point types (`fixedN_M` / `ufixedN_M`) — see
//! `docs/fixed-point-types.md` and `src/fixed.rs`.
//!
//! The Rust side is a generated newtype per used format (`Fixed16_16(pub i32)`)
//! whose operator impls carry the semantics. What the transpiler adds is the part
//! Rust cannot infer: an *untyped* numeric literal next to a fixed-point operand
//! (`x + 0.25`, `let fixed16_16 y = 3`) is folded to the raw integer at transpile
//! time, and `as` casts between fixed-point and int/float go through the
//! `__FixedCast` / `__FixedFormat` traits.

use super::*;
use super::emit_expr::is_known_numeric_scalar_type;
use crate::fixed::FixedFmt;

impl Transpiler {
    /// The fixed-point format a declared type names, looking through `mut` and
    /// ownership qualifiers.
    pub(crate) fn fixed_fmt_of_type(ty: &Type) -> Option<FixedFmt> {
        match ty {
            Type::Fixed(f) => Some(*f),
            Type::Named(n) => FixedFmt::from_name(n),
            Type::Mut(inner) | Type::Qualified(inner, _) => Self::fixed_fmt_of_type(inner),
            _ => None,
        }
    }

    /// Static fixed-point format of an expression, when known. Arithmetic takes the
    /// format of whichever operand has one (Boring requires both to agree).
    pub(crate) fn fixed_fmt_of_expr(&self, e: &Expr) -> Option<FixedFmt> {
        self.fixed_fmt_of_expr_inner(e, &mut std::collections::HashSet::new())
    }

    /// `visiting` guards the `var_init_exprs` fallback (a flat name -> latest
    /// initializer map) against self-referential shadowing (`let x = x * 2`).
    fn fixed_fmt_of_expr_inner(&self, e: &Expr, visiting: &mut std::collections::HashSet<String>) -> Option<FixedFmt> {
        match &e.kind {
            ExprKind::Var(v) => {
                if let Some(f) = self.resolve_expr_type(e).and_then(|t| Self::fixed_fmt_of_type(&t)) {
                    return Some(f);
                }
                // An unannotated `let big = 3 as fixed32_32` isn't tracked into `var_types`.
                if !visiting.insert(v.clone()) {
                    return None;
                }
                let r = self.var_init_exprs.get(v.as_str()).and_then(|init| self.fixed_fmt_of_expr_inner(init, visiting));
                visiting.remove(v.as_str());
                r
            }
            ExprKind::Cast(_, ty) => Self::fixed_fmt_of_type(ty),
            ExprKind::BinOp(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem, l, r) =>
                self.fixed_fmt_of_expr_inner(l, visiting).or_else(|| self.fixed_fmt_of_expr_inner(r, visiting)),
            ExprKind::UnaryOp(UnaryOp::Neg, inner) => self.fixed_fmt_of_expr_inner(inner, visiting),
            // Members returning the receiver's own format (`x.sin()`, `x.min(y)`, `x.wrappingAdd(y)`).
            // `checked*` return an optional and are handled where they are unwrapped.
            ExprKind::MethodCall(obj, m, _) if matches!(m.as_str(),
                "floor" | "ceil" | "round" | "trunc" | "fract" | "abs" | "signum" | "min" | "max" | "clamp"
                | "sqrt" | "sin" | "cos" | "tan" | "exp" | "ln" | "mulRound" | "divRound")
                || m.starts_with("wrapping") || m.starts_with("saturating") =>
                self.fixed_fmt_of_expr_inner(obj, visiting),
            _ => self.resolve_expr_type(e).and_then(|t| Self::fixed_fmt_of_type(&t)),
        }
    }

    /// `Some(raw)` for an untyped numeric literal (optionally negated). The outer
    /// `Option` is "is this a literal"; the inner one is "does it fit `fmt`".
    fn fixed_literal_raw(e: &Expr, fmt: &FixedFmt) -> Option<Option<i128>> {
        match &e.kind {
            ExprKind::Int(n) => Some(fmt.from_int(*n as i128)),
            ExprKind::Float(f) => Some(fmt.from_f64_literal(*f)),
            ExprKind::UnaryOp(UnaryOp::Neg, inner) => match &inner.kind {
                ExprKind::Int(n) => Some(fmt.from_int(-(*n as i128))),
                ExprKind::Float(f) => Some(fmt.from_f64_literal(-*f)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Emits an untyped numeric literal as a `fmt` constant, or `None` when `e` is
    /// not such a literal. An out-of-range literal is a transpile error.
    pub(crate) fn emit_fixed_literal(&self, e: &Expr, fmt: &FixedFmt) -> Option<String> {
        let res = Self::fixed_literal_raw(e, fmt)?;
        Some(match res {
            Some(raw) => format!("{}::from_raw({})", fmt.rust_name(), raw),
            None => {
                self.push_error(e.line, e.col, format!(
                    "literal out of range for {} (range [{}, {}])",
                    fmt.name(), fmt.format_exact(fmt.min_raw()), fmt.format_exact(fmt.max_raw())));
                format!("{}::default()", fmt.rust_name())
            }
        })
    }

    /// True when `ty` is a fixed-point type or a collection/tuple nesting one.
    fn type_mentions_fixed(ty: &Type) -> bool {
        match ty {
            Type::Array(t) | Type::ArrayN(t, _) | Type::Set(t) => Self::type_mentions_fixed(t),
            Type::Dict(k, v) => Self::type_mentions_fixed(k) || Self::type_mentions_fixed(v),
            Type::Tuple(ts) => ts.iter().any(Self::type_mentions_fixed),
            Type::Mut(t) | Type::Qualified(t, _) => Self::type_mentions_fixed(t),
            _ => Self::fixed_fmt_of_type(ty).is_some(),
        }
    }

    /// `[1.5, 2.25]` against `[fixed16_16]` (also `{K = fixedN_M}`, tuples, sets and
    /// nestings): folds every untyped literal element to the element format.
    /// `None` when `ty` holds no fixed-point type or `value` is not a literal collection.
    pub(crate) fn emit_fixed_collection_literal(&self, ty: &Type, value: &Expr) -> Option<String> {
        let ty = ty.without_mut();
        if !Self::type_mentions_fixed(ty) {
            return None;
        }
        match (ty, &value.kind) {
            (Type::Array(et), ExprKind::Array(elems)) if !elems.is_empty() => {
                let es: Vec<String> = elems.iter().map(|e| self.emit_let_value(Some(et), e)).collect();
                Some(format!("vec![{}]", es.join(", ")))
            }
            (Type::ArrayN(et, _), ExprKind::Array(elems)) if !elems.is_empty() => {
                let es: Vec<String> = elems.iter().map(|e| self.emit_let_value(Some(et), e)).collect();
                Some(format!("[{}]", es.join(", ")))
            }
            (Type::Set(et), ExprKind::Set(elems)) if !elems.is_empty() => {
                let es: Vec<String> = elems.iter().map(|e| self.emit_let_value(Some(et), e)).collect();
                Some(format!("HashSet::from([{}])", es.join(", ")))
            }
            (Type::Tuple(ts), ExprKind::Tuple(elems)) if ts.len() == elems.len() => {
                let es: Vec<String> = ts.iter().zip(elems).map(|(t, e)| self.emit_let_value(Some(t), e)).collect();
                Some(format!("({})", es.join(", ")))
            }
            (Type::Dict(kt, vt), ExprKind::Dict(pairs)) if !pairs.is_empty() => {
                let ps: Vec<String> = pairs.iter()
                    .map(|(k, v)| format!("({}, {})", self.emit_let_value(Some(kt), k), self.emit_let_value(Some(vt), v)))
                    .collect();
                Some(format!("HashMap::from([{}])", ps.join(", ")))
            }
            _ => None,
        }
    }

    /// Operand of a fixed-point operation: a bare literal takes the operation's
    /// format, anything else is emitted normally.
    fn emit_fixed_operand(&self, e: &Expr, fmt: &FixedFmt) -> String {
        self.emit_fixed_literal(e, fmt).unwrap_or_else(|| self.emit_expr(e))
    }

    /// `x.floor()`, `x.min(y)`, `fixed16_16.fromRaw(n)`, ... on a fixed-point receiver
    /// (or the fixed-point type name itself).
    pub(crate) fn try_emit_fixed_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        if let ExprKind::Var(v) = &obj.kind {
            if let Some(fmt) = FixedFmt::from_name(v) {
                return match (method, args) {
                    ("fromRaw", [a]) => Some(format!("{}::from_raw(({}) as {})", fmt.rust_name(), self.emit_expr(&a.value), fmt.rust_int())),
                    _ => None,
                };
            }
        }
        let rust_method = match method {
            "floor" | "ceil" | "round" | "trunc" | "fract" | "abs" | "signum" | "min" | "max" | "clamp"
            | "sqrt" | "sin" | "cos" | "tan" | "exp" | "ln" => method.to_string(),
            "mulRound" => "mul_round".to_string(),
            "divRound" => "div_round".to_string(),
            // wrappingAdd -> wrapping_add, checkedDiv -> checked_div, ...
            m => {
                let (policy, op) = ["wrapping", "saturating", "checked"].iter()
                    .find_map(|p| m.strip_prefix(p).map(|r| (*p, r)))?;
                if !matches!(op, "Add" | "Sub" | "Mul" | "Div") {
                    return None;
                }
                format!("{}_{}", policy, op.to_lowercase())
            }
        };
        let fmt = self.fixed_fmt_of_expr(obj)?;
        let args_s: Vec<String> = args.iter().map(|a| self.emit_fixed_operand(&a.value, &fmt)).collect();
        Some(format!("{}.{}({})", self.emit_expr(obj), rust_method, args_s.join(", ")))
    }

    /// `<fixed>.checkedOp(y) else default` -> `.unwrap_or(default)` with a literal
    /// default folded to the receiver's format.
    pub(crate) fn try_emit_fixed_else(&self, e: &Expr, default: &Expr) -> Option<String> {
        let ExprKind::MethodCall(obj, method, _) = &e.kind else { return None };
        if !method.starts_with("checked") {
            return None;
        }
        let fmt = self.fixed_fmt_of_expr(obj)?;
        let call = self.try_emit_fixed_method(obj, method, match &e.kind { ExprKind::MethodCall(_, _, a) => a, _ => return None })?;
        Some(format!("{}.unwrap_or({})", call, self.emit_fixed_operand(default, &fmt)))
    }

    /// `x.raw`, `fixed16_16.epsilon` / `.min` / `.max`.
    pub(crate) fn try_emit_fixed_field(&self, obj: &Expr, field: &str) -> Option<String> {
        if let ExprKind::Var(v) = &obj.kind {
            if let Some(fmt) = FixedFmt::from_name(v) {
                let c = match field { "epsilon" => "EPSILON", "min" => "MIN", "max" => "MAX", _ => return None };
                return Some(format!("{}::{}", fmt.rust_name(), c));
            }
        }
        if field == "raw" && self.fixed_fmt_of_expr(obj).is_some() {
            return Some(format!("{}.raw()", self.emit_expr(obj)));
        }
        None
    }

    /// `l op r` where at least one side is statically fixed-point.
    pub(crate) fn try_emit_fixed_binop(&self, op: &BinOp, l: &Expr, r: &Expr) -> Option<String> {
        let sym = match op {
            BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*", BinOp::Div => "/", BinOp::Rem => "%",
            BinOp::Eq => "==", BinOp::NotEq => "!=", BinOp::Lt => "<", BinOp::Gt => ">",
            BinOp::LtEq => "<=", BinOp::GtEq => ">=",
            _ => return None,
        };
        let fmt = self.fixed_fmt_of_expr(l).or_else(|| self.fixed_fmt_of_expr(r))?;
        // `x == nil` and friends are handled by the generic path.
        if matches!(l.kind, ExprKind::Nil) || matches!(r.kind, ExprKind::Nil) {
            return None;
        }
        Some(format!("({} {} {})", self.emit_fixed_operand(l, &fmt), sym, self.emit_fixed_operand(r, &fmt)))
    }

    /// `e as T` when `e` or `T` is fixed-point.
    pub(crate) fn try_emit_fixed_cast(&self, e: &Expr, ty: &Type) -> Option<String> {
        let src_fmt = self.fixed_fmt_of_expr(e);
        if let Some(to) = Self::fixed_fmt_of_type(ty) {
            // A literal operand is the decimal the author wrote: round to nearest.
            if let Some(lit) = self.emit_fixed_literal(e, &to) {
                return Some(lit);
            }
            return Some(format!(
                "__FixedCast::__to_fixed::<{}>({}).expect(\"fixed-point cast out of range\")",
                to.rust_name(), self.emit_expr(e)));
        }
        src_fmt?;
        let src = self.emit_expr(e);
        let dst = self.emit_type(ty);
        match ty.without_mut() {
            t if is_known_numeric_scalar_type(t) && dst.starts_with('f') =>
                Some(format!("(__FixedFormat::__to_f64({}) as {})", src, dst)),
            t if is_known_numeric_scalar_type(t) =>
                Some(format!("(__FixedFormat::__to_i128({}) as {})", src, dst)),
            _ => None,
        }
    }

}

/// Prelude items for every fixed-point format that appears in the generated
/// code, to append to the program, plus an error message per unsupported
/// (128-bit) format. A free function: it runs after the transpiler's output has
/// been moved out of `Transpiler`.
pub(crate) fn fixed_point_prelude(generated: &str) -> (Option<String>, Vec<String>) {
    {
        let mut fmts: Vec<FixedFmt> = Vec::new();
        let bytes = generated.as_bytes();
        let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
        // Every whole identifier `Fixed<n>_<m>` / `UFixed<n>_<m>` (ASCII-only, so the
        // byte offsets from `match_indices` are always char boundaries).
        for (at, _) in generated.match_indices("Fixed") {
            let start = if at > 0 && bytes[at - 1] == b'U' { at - 1 } else { at };
            if start > 0 && is_ident(bytes[start - 1]) {
                continue;
            }
            let end = bytes[at..].iter().position(|&b| !is_ident(b)).map_or(bytes.len(), |o| at + o);
            let word = &generated[start..end];
            let lower = if start < at { word.replacen("UFixed", "ufixed", 1) } else { word.replacen("Fixed", "fixed", 1) };
            if let Some(fmt) = FixedFmt::from_name(&lower) {
                if fmt.rust_name() == word && !fmts.contains(&fmt) {
                    fmts.push(fmt);
                }
            }
        }
        if fmts.is_empty() {
            return (None, Vec::new());
        }
        let mut errors = Vec::new();
        fmts.sort_by_key(|f| (f.width(), !f.signed, f.n));
        let mut out = String::from(crate::fixed::RUST_SHARED_PRELUDE);
        for f in &fmts {
            if f.width() > 64 {
                errors.push(format!(
                    "{} is not supported yet: 128-bit fixed-point formats need a 256-bit intermediate \
                     for `*` and `/` (docs/fixed-point-types.md, Phasing)", f.name()));
                continue;
            }
            out.push_str(&f.rust_definition());
        }
        (Some(out), errors)
    }
}
        // Integer math core (sqrt/sin/cos/tan/exp/ln), the very file the Rust interpreter compiles.
        out.push_str("#[allow(dead_code)]\nmod __fxm {\n");
        out.push_str(include_str!("../fixed_math_core.rs"));
        out.push_str("}\n");
