use super::*;

/// Boring's built-in numeric methods (`x.exp()`, `x.sqrt()`, `x.tanh()`, ...) as
/// CUDA C / HIP C++ device code -- both toolchains expose the same standard C
/// math library function names in device code (`sqrt`, `exp`, `tanh`, etc.), so
/// this is shared verbatim between the two backends. Mirrors `metal::device`'s
/// `float_unary_method_msl`, adapted to C names (`fabs` not `abs`, no Metal-only
/// `M_PI_F`/`M_E_F` macros -- literal constants instead, portable across both).
/// Returns `None` for anything that isn't one of Boring's built-in float methods,
/// so callers can fall through to their own "unsupported method" marker.
///
/// Without this, `(v + 1e-5).sqrt()`-style method-call syntax (as opposed to the
/// free-function form `sqrt(v + 1e-5)`, already handled by `map_builtin_fn`) fell
/// through to the generic "no C equivalent" marker, which -- unlike a marker on a
/// full statement -- silently produced a syntactically invalid *expression*
/// (`const auto e = /* unsupported: ... */;`, missing a value between `=` and
/// `;`). Confirmed via whisper-boring's `src/math_gpu.br` (softmax's `.exp()`,
/// layernorm's `.sqrt()`, gelu's `.tanh()`) targeting `--target cuda`.
pub(crate) fn float_unary_method_c(method: &str, obj: &str, args: &[String]) -> Option<String> {
    let simple = match method {
        "sqrt" => "sqrt", "cbrt" => "cbrt", "abs" => "fabs",
        "floor" => "floor", "ceil" => "ceil", "round" => "round",
        "exp" => "exp", "exp2" => "exp2", "ln" => "log",
        "log2" => "log2", "log10" => "log10",
        "sin" => "sin", "cos" => "cos", "tan" => "tan",
        "asin" => "asin", "acos" => "acos", "atan" => "atan",
        "sinh" => "sinh", "cosh" => "cosh", "tanh" => "tanh",
        _ => "",
    };
    if !simple.is_empty() {
        return Some(format!("{}({})", simple, obj));
    }
    match method {
        "pow" | "powf" => {
            let exp = args.first().cloned().unwrap_or_else(|| "1.0".into());
            Some(format!("pow({}, {})", obj, exp))
        }
        "log" => {
            let base = args.first().cloned().unwrap_or_else(|| "2.718281828459045".into());
            Some(format!("(log({}) / log({}))", obj, base))
        }
        "atan2" => {
            let other = args.first().cloned().unwrap_or_else(|| "0.0".into());
            Some(format!("atan2({}, {})", obj, other))
        }
        "signum" => Some(format!("copysign(1.0, {})", obj)),
        "recip"  => Some(format!("(1.0 / {})", obj)),
        "toRadians" => Some(format!("({} * (3.14159265358979323846 / 180.0))", obj)),
        "toDegrees" => Some(format!("({} * (180.0 / 3.14159265358979323846))", obj)),
        _ => None,
    }
}

/// True when the RHS of a top-level `let name = val` (with optional type
/// annotation `ty`) is a scalar constant suitable for inlining into GPU device/host
/// code as a `top_level_scalars` entry (see that field's doc across the
/// cuda/rocm/metal/wgpu backends' host.rs/device.rs). Handles unary-negated literals
/// (`let x_min = -2.0`) -- a bare `Int(_)|Float(_)|Bool(_)` check on `val.kind` misses
/// these, since unary minus wraps the literal in `UnaryOp(Neg, ...)`, not a literal
/// itself. Confirmed via examples/mandelbrot_gpu.br's `let x_min = -2.0` / `let y_min
/// = -1.5`: silently failed to inline, reaching generated CUDA device code as
/// undefined identifiers (`x_min`, `y_min`) while the positive constants on the same
/// lines (`x_max`, `y_max`, `width`, `height`) inlined fine.
pub(crate) fn is_scalar_let_value(val: &Expr, ty: Option<&Type>) -> bool {
    fn is_scalar_literal(e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) => true,
            ExprKind::UnaryOp(UnaryOp::Neg, inner) => is_scalar_literal(inner),
            _ => false,
        }
    }
    is_scalar_literal(val)
        || ty.map(|t| matches!(t, Type::Int | Type::Uint | Type::Float32 | Type::Float64 | Type::Bool)).unwrap_or(false)
}

/// Rust source for boring's built-in `Dimension` type, used by 2-D kernels.
/// Emitted verbatim by both the cuda and metal host backends.
pub(crate) const DIMENSION_STRUCT_RUST: &str =
    "#[repr(C)]\n\
     #[derive(Copy, Clone, Debug)]\n\
     struct Dimension { width: u32, height: u32 }\n\
     #[allow(non_snake_case)]\n\
     fn Dimension(width: u32, height: u32) -> Dimension { Dimension { width, height } }";

/// The three ceil-div grid-axis expressions for a desugared dynamic-shape
/// field's shadow fields (see `desugared_labeled_array_shadow_fields`),
/// reading each shadow's *runtime* value through `{receiver}.{shadow}`
/// instead of a `ConstInt` literal — the one real difference from
/// `labeled_array_grid_dim_expr`'s fixed-shape formula, which this otherwise
/// mirrors exactly (same ceil-div shape, same "missing axis defaults to 1").
///
/// `receiver` is `"self"` for CUDA/ROCm/Metal's `__boring_launch` method
/// context (reading a sibling field on the same struct) or a kernel-instance
/// variable name for wgpu's dispatch-call-site context (`transpiler::
/// emit_kernel`'s `try_emit_kernel_dispatch`, which computes the grid at the
/// `kernel: k(...)` call site itself, not inside a method on `k`'s type) —
/// see that module for why wgpu's shape differs from the other three.
/// `block_axes` are the three block-dim expressions to ceil-divide against,
/// in `[x, y, z]` order (`["block_dim.0", "block_dim.1", "block_dim.2"]` for
/// CUDA/ROCm/Metal's `(u32,u32,u32)` struct field; the dispatch call's own
/// per-axis `block=` arg expressions for wgpu).
pub(crate) fn shadow_grid_axes(receiver: &str, shadows: &[String], block_axes: [&str; 3]) -> (String, String, String) {
    let axis_expr = |i: usize| -> String {
        match shadows.get(i) {
            Some(s) => {
                let ax = block_axes[i];
                format!("((({receiver}.{s}) as u32 + ({ax}) - 1) / ({ax}))")
            }
            None => "1".to_string(),
        }
    };
    (axis_expr(0), axis_expr(1), axis_expr(2))
}

// ─── Labeled multi-dimensional arrays (docs/array-multidim-types.md) ───────
//
// A `LabeledAxis`'s fixed size is an arbitrary `ConstExpr` (may reference a
// kernel const-generic param, e.g. `width = W`), not always a literal
// `ConstInt` — `const_expr_to_c_like` below stringifies that general case.

/// Stringifies a `LabeledAxis`'s fixed-size `ConstExpr` into the infix
/// arithmetic syntax shared by all 4 GPU backends — CUDA C, HIP C++, Metal
/// MSL, and WGSL all use the same `+`/`-`/`*`/`/`/parens/unary-minus syntax
/// for integer arithmetic, and a bare Boring identifier is already valid
/// syntax in each, so one shared stringifier covers every backend. Covers
/// exactly the shapes `spec/grammar.bnf`'s `const_expr` production allows
/// (`INT | IDENT | const_expr (+|-|*|/) const_expr | -const_expr |
/// (const_expr)`) — the same restricted grammar `[T, N]`'s `N` already
/// uses. The fallback marker should be unreachable: the checker rejects a
/// non-const-expr axis size before this ever runs.
pub(crate) fn const_expr_to_c_like(e: &Expr) -> String {
    match &e.kind {
        ExprKind::Int(n) => n.to_string(),
        ExprKind::Var(name) => name.clone(),
        ExprKind::UnaryOp(UnaryOp::Neg, inner) => format!("(-{})", const_expr_to_c_like(inner)),
        ExprKind::BinOp(op, l, r) => {
            let sym = match op {
                BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*", BinOp::Div => "/",
                _ => return "/* unsupported const expr */".to_string(),
            };
            format!("({} {} {})", const_expr_to_c_like(l), sym, const_expr_to_c_like(r))
        }
        _ => "/* unsupported const expr */".to_string(),
    }
}

fn const_expr_to_c_like_axis(axis: &crate::ast::LabeledAxis) -> String {
    let crate::ast::ConstExpr(boxed) = axis.size.as_ref()
        .expect("fixed-shape LabeledArray: every axis has Some(size) by construction");
    const_expr_to_c_like(boxed)
}

/// Product of `axes`' sizes, as a string — folds to a single literal (e.g.
/// `"480000"`) when every axis size is a literal int; falls back to a
/// `*`-joined expression string (e.g. `"W * H"`) the moment any axis
/// references a const-generic param.
fn labeled_array_stride_str(axes: &[crate::ast::LabeledAxis]) -> String {
    let mut product: Option<i64> = Some(1);
    for a in axes {
        let lit = a.size.as_ref().and_then(|crate::ast::ConstExpr(boxed)| match &boxed.kind {
            ExprKind::Int(n) => Some(*n),
            _ => None,
        });
        product = match (product, lit) {
            (Some(p), Some(n)) => Some(p * n),
            _ => None,
        };
        if product.is_none() { break; }
    }
    match product {
        Some(n) => n.to_string(),
        None => axes.iter().map(const_expr_to_c_like_axis).collect::<Vec<_>>().join(" * "),
    }
}

/// Row-major flat-buffer index for a fixed-shape `LabeledArray`'s
/// `LabeledIndex` (`a[width = w, height = h]`). `axes` are the type's axes
/// (every one fixed — only a fixed-shape field ever reaches this, dynamic
/// shape is desugared away before codegen); `args` are (label,
/// already-stringified index expr) pairs, in whatever order the call site
/// wrote them (order is free at the use site — see the design doc). `None`
/// if any axis's label isn't found among `args` (should be unreachable
/// post-checker).
pub(crate) fn labeled_array_at_index(axes: &[crate::ast::LabeledAxis], args: &[(String, String)]) -> Option<String> {
    let mut terms: Vec<String> = Vec::with_capacity(axes.len());
    for (i, axis) in axes.iter().enumerate() {
        let (_, idx_str) = args.iter().find(|(label, _)| label == &axis.label)?;
        if i == 0 {
            terms.push(idx_str.clone());
        } else {
            terms.push(format!("{} * {}", idx_str, labeled_array_stride_str(&axes[..i])));
        }
    }
    Some(terms.join(" + "))
}

/// Stringified value of a fixed-shape `LabeledArray`'s axis, by label — for
/// `a.axis`'s shape-query property lowering. `None` if `axis_label` doesn't
/// match any of `axes`.
pub(crate) fn labeled_array_dim_literal(axes: &[crate::ast::LabeledAxis], axis_label: &str) -> Option<String> {
    let axis = axes.iter().find(|a| a.label == axis_label)?;
    let crate::ast::ConstExpr(boxed) = axis.size.as_ref()?;
    Some(const_expr_to_c_like(boxed))
}

/// Ceil-div grid-dim tuple expression for a fixed-shape `LabeledArray`
/// field, generalized to N axes. Missing axes (fewer than 3 declared)
/// default to `1`, same as the existing 1D/2D/3D cases.
pub(crate) fn labeled_array_grid_dim_expr(axes: &[crate::ast::LabeledAxis]) -> String {
    let block_axis = |i: usize| match i { 0 => "block_dim.0", 1 => "block_dim.1", _ => "block_dim.2" };
    let axis_expr = |i: usize| -> String {
        match axes.get(i).and_then(|a| a.size.as_ref()) {
            Some(crate::ast::ConstExpr(boxed)) => {
                let n = const_expr_to_c_like(boxed);
                format!("(({} + {ax} - 1) / {ax})", n, ax = block_axis(i))
            }
            None => "1".to_string(),
        }
    };
    format!("({}, {}, {})", axis_expr(0), axis_expr(1), axis_expr(2))
}

/// Detects a *desugared* dynamic-shape `LabeledArray` field's shadow
/// siblings, using `desugar_labeled_array`'s
/// positional naming (`__{field}_axis0`/`_axis1`/`_axis2`, not label-text-
/// based — labels are arbitrary user text). Returns them in axis order;
/// `None` for a plain dynamic array with no such shadow siblings. Feeds
/// directly into the existing `shadow_grid_axes` (already name-agnostic —
/// no LabeledArray-specific sibling needed there).
pub(crate) fn desugared_labeled_array_shadow_fields(field_name: &str, all_fields: &[KernelFieldDecl]) -> Option<Vec<String>> {
    let shadow = |i: usize| -> Option<String> {
        let name = format!("__{field_name}_axis{i}");
        all_fields.iter().any(|f| f.name == name).then_some(name)
    };
    let axis0 = shadow(0)?;
    let axis1 = shadow(1)?;
    let mut out = vec![axis0, axis1];
    if let Some(axis2) = shadow(2) {
        out.push(axis2);
    }
    Some(out)
}

pub(crate) fn looks_like_collection(expr: &str) -> bool {
    // `arr.join(sep)`/`arr.joined(sep)` always collapses a collection into a single
    // owned `String` (`.iter().map(...).collect::<Vec<&str>>().join(&*sep)` — see
    // emit_methods.rs's `"joined" | "join"` case) — never a collection itself, no
    // matter that the emitted expression's *prefix* happens to start with `vec![`
    // (the array literal being joined). Must be checked before the `starts_with`
    // heuristic below, which only looks at the front of the string and would
    // otherwise treat the joined String as a Vec and wrongly Debug-quote it in a
    // `print`/string interpolation (confirmed via examples/hello.br: `print "join:
    // {["a","b","c"].join(", ")}"` rendered `"a, b, c"` with quotes instead of
    // `a, b, c`).
    if expr.contains(".collect::<Vec<&str>>().join(") {
        return false;
    }
    // Subscript access on a collection yields an element, not a collection.
    // E.g. `arr.collect::<Vec<_>>()[0].clone()` is a scalar, not a Vec.
    let has_vec_collect = expr.contains(".collect::<Vec<_>>()")
        || expr.contains(".collect::<HashMap")
        || expr.contains(".collect::<HashSet");
    if has_vec_collect {
        // Find the last .collect occurrence and check if a `[` follows it.
        let is_subscripted = [".collect::<Vec<_>>()", ".collect::<HashMap", ".collect::<HashSet"]
            .iter()
            .filter_map(|pat| expr.rfind(pat).map(|p| p + pat.len()))
            .any(|after| expr[after..].contains('['));
        if is_subscripted { return false; }
    }
    has_vec_collect ||
    expr.starts_with("vec![") ||
    expr.starts_with("Vec::") ||
    expr.starts_with("HashMap::") ||
    expr.starts_with("HashSet::")
}

/// Returns true if `expr` looks like a HashMap or HashSet collection (not Vec).
/// These don't implement Display, so `{:?}` must be used for them.
pub(crate) fn looks_like_map_or_set(expr: &str) -> bool {
    let has_map_collect = expr.contains(".collect::<HashMap") || expr.contains(".collect::<HashSet");
    if has_map_collect {
        let is_subscripted = [".collect::<HashMap", ".collect::<HashSet"]
            .iter()
            .filter_map(|pat| expr.rfind(pat).map(|p| p + pat.len()))
            .any(|after| expr[after..].contains('['));
        if is_subscripted { return false; }
        return true;
    }
    expr.starts_with("HashMap::") || expr.starts_with("HashSet::")
}

/// Returns true when an expression string clearly resolves to a Vec<T> (not a scalar).
/// Used to decide whether BoringFmt wrapping is safe.
/// Conservative: only matches expressions that END as a Vec (starts with `vec![` or
/// ends with `.collect::<Vec<_>>()`).  Method-chains ending in `.fold()`/`.count()` etc.
/// are scalars even if they contain an intermediate `.collect::<Vec<_>>()` step.
pub(crate) fn expr_ends_as_vec(expr: &str) -> bool {
    let trimmed = expr.trim_end();
    // Pure vec![ ] literal: no method chain follows the closing bracket.
    // `vec![1, 2, 3]` → true; `vec![1,2].iter().fold(0, ...)` → false (ends with `)`)
    if trimmed.starts_with("vec![") && trimmed.ends_with(']') {
        return true;
    }
    // Method chain ending as a collected Vec.
    if trimmed.ends_with(".collect::<Vec<_>>()") { return true; }
    // Block expression whose last statement is a Vec collect
    // e.g. `{ let mut __boring_v = ...; __boring_v.sort_by(...); __boring_v.iter().cloned().collect::<Vec<_>>() }`
    if trimmed.starts_with('{') && trimmed.ends_with(".collect::<Vec<_>>() }") { return true; }
    false
}

/// Wraps a Vec expression in `BoringFmt(&...)` and returns the `{}` spec.
/// For HashMap/HashSet, keeps the expression as-is with `{:?}`.
/// For ambiguous collection vars (e.g. reduce results in collection_vars), falls
/// back to `{:?}` so scalars still compile.
/// `is_vec_var` is true when the emitted expression is a variable known to be in `vec_vars`.
/// Returns `(possibly_wrapped_expr, format_spec)`.
pub(crate) fn boring_vec_fmt(expr: String, is_col: bool, is_vec_var: bool) -> (String, &'static str) {
    if !is_col { return (expr, "{}"); }
    if looks_like_map_or_set(&expr) { return (expr, "{:?}"); }
    // Wrap with BoringFmt when:
    // 1. Expression unambiguously ends as a Vec (inline collect/vec![...])
    // 2. Variable is tracked in vec_vars (assigned from a clear Vec expression)
    if expr_ends_as_vec(&expr) || is_vec_var {
        (format!("BoringFmt(&{})", expr), "{}")
    } else {
        // Var in collection_vars but we can't be sure — keep {:?} (safe for scalars).
        (expr, "{:?}")
    }
}

/// Collect all lifetime letters used in a type, recursively.
/// E.g. `Qualified(Str, Lifetime("a"))` → `["a"]`.
pub(crate) fn collect_lifetimes(ty: &Type, out: &mut Vec<String>) {
    match ty {
        Type::Qualified(inner, OwnerQual::Lifetime(lt)) => {
            if !out.contains(lt) { out.push(lt.clone()); }
            collect_lifetimes(inner, out);
        }
        Type::Qualified(inner, _) => collect_lifetimes(inner, out),
        Type::Optional(inner) | Type::Array(inner) | Type::Set(inner) | Type::Dyn(inner) | Type::Impl(inner) => {
            collect_lifetimes(inner, out);
        }
        Type::Dict(k, v) => { collect_lifetimes(k, out); collect_lifetimes(v, out); }
        Type::Tuple(elems) => elems.iter().for_each(|t| collect_lifetimes(t, out)),
        Type::Generic(_, args) => args.iter().for_each(|t| collect_lifetimes(t, out)),
        Type::Fn(ret, params, _, _, _) => {
            if let Some(r) = ret { collect_lifetimes(r, out); }
            params.iter().for_each(|t| collect_lifetimes(t, out));
        }
        // Bare lifetime stored as Named("'a") from parse_generic_type_arg
        Type::Named(s) if s.starts_with('\'') => {
            let lt = s[1..].to_string();
            if !out.contains(&lt) { out.push(lt); }
        }
        Type::AssocOf(base, _) => collect_lifetimes(base, out),
        _ => {}
    }
}

/// Does an explicit type annotation indicate a collection?
pub(crate) fn is_collection_type(ty: Option<&Type>) -> bool {
    match ty {
        Some(Type::Array(_)) | Some(Type::Dict(_, _)) | Some(Type::Set(_)) => true,
        Some(Type::Named(n)) => matches!(n.as_str(), "Vec" | "HashMap" | "HashSet" | "BTreeMap" | "BTreeSet"),
        Some(Type::Generic(n, _)) => matches!(n.as_str(), "Vec" | "HashMap" | "HashSet"),
        _ => false,
    }
}

/// Boring-side method names on `[T]` / `{K=V}` / `{T}` that require `&mut` access on their
/// receiver (they mutate the collection in place). Every other built-in collection method
/// (`len`, `contains`, `map`, `first`, …) is read-only. Used both to gate write access on
/// `'actor`-qualified fields and to decide whether a bare array/dict/set parameter can be
/// passed by reference (see infer_qualifiers.rs) instead of cloned.
pub(crate) const MUTATING_COLLECTION_METHODS: &[&str] = &[
    "append", "add", "push", "extend", "insert", "set", "remove", "removeAt", "remove_at",
    "pop", "clear", "sort", "sortBy", "sort_by", "reverse", "shuffle", "dedup",
    "retain", "truncate", "drain",
];

/// Normalize boring primitive type names (lowercase aliases) to Rust equivalents.
/// Pass `use_rc = true` in single-thread mode so `string` maps to `Rc<str>` instead of `Arc<str>`.
pub(crate) fn normalize_type_name(name: &str, use_rc: bool) -> String {
    match name {
        "string"            => if use_rc { "Rc<str>".into() } else { "Arc<str>".into() },
        "str"               => "&str".into(),
        "String"            => "String".into(),
        "int"    | "Int"    => "isize".into(),
        "uint"   | "Uint"   => "usize".into(),
        "uint8"  | "Uint8"  => "u8".into(),
        "int8"    | "Int8"    => "i8".into(),
        "int16"   | "Int16"   => "i16".into(),
        "int32"   | "Int32"   => "i32".into(),
        "int64"   | "Int64"   => "i64".into(),
        "int128"  | "Int128"  => "i128".into(),
        "uint16"  | "Uint16"  => "u16".into(),
        "uint32"  | "Uint32"  => "u32".into(),
        "uint64"  | "Uint64"  => "u64".into(),
        "uint128" | "Uint128" => "u128".into(),
        // `float`/`Float` are pure aliases of `float64` (docs/float-width-types.md §2) —
        // fold into the same "f64" bucket "float64"/"Float64" resolve to.
        "float32" | "Float32" => "f32".into(),
        "float"   | "Float" | "float64" | "Float64" => "f64".into(),
        "bool"   | "Bool"   => "bool".into(),
        "void"   | "Void"   => "()".into(),
        "nil"    | "Nil"    => "()".into(),
        "never"  | "Never"  => "!".into(),
        // Rust numeric aliases pass through unchanged
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" => name.into(),
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" => name.into(),
        "f32" | "f64" => name.into(),
        // Qualify stdlib module paths that may not be in scope
        other if other.starts_with("io::") => format!("std::{}", other),
        other => other.into(),
    }
}

pub(crate) fn binop_str(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add    => "+",
        BinOp::Sub    => "-",
        BinOp::Mul    => "*",
        BinOp::Div    => "/",
        BinOp::Rem    => "%",
        BinOp::Eq     => "==",
        BinOp::RefEq  => "==",   // unreachable — handled as Arc::ptr_eq in emit_expr
        BinOp::NotEq  => "!=",
        BinOp::Lt     => "<",
        BinOp::Gt     => ">",
        BinOp::LtEq   => "<=",
        BinOp::GtEq   => ">=",
        BinOp::And    => "&&",
        BinOp::Or     => "||",
        BinOp::BitAnd => "&",
        BinOp::BitOr  => "|",
        BinOp::BitXor => "^",
        BinOp::Shl    => "<<",
        BinOp::Shr    => ">>",
        BinOp::Is     => "==",   // approximate; proper type checking needed
        BinOp::IsNot  => "!=",
    }
}

/// True when `e` is a bare, argument-less `.pop()` call (`items.pop()`, not chained
/// further) — the shape a `T?`-returning function/`let` needs to recognize so it can pass
/// `Vec::pop()`'s `Option<T>` through raw instead of letting `map_method` unwrap it (see
/// `map_method`'s `want_raw_option` doc) and then re-wrapping it in `Some(...)`.
pub(crate) fn is_bare_pop_call(e: &Expr) -> bool {
    matches!(&e.kind, ExprKind::MethodCall(_, m, a) | ExprKind::Pipe(_, m, a) if m == "pop" && a.is_empty())
}

/// Map boring method names to (rust_method, optional_suffix).
///
/// `want_raw_option` — true only while emitting an expression flowing directly into an
/// Optional-typed function return / `let T?` binding (`Transpiler::want_raw_option_pop`,
/// set narrowly around that outer expression — see its doc). It affects only "pop": when
/// true, the `.unwrap_or_default()` suffix below is skipped so the raw `Option<T>` Rust's
/// `Vec::pop()` produces flows through unchanged, matching array.br's own declared
/// `req T? pop(): native` signature (nil if empty) instead of discarding `None`. In every
/// other (non-Optional-target) context — the common case, e.g. `let v = arr.pop()` with
/// `v` inferred as a bare `T` — the suffix stays on, since that's the position Boring's
/// "returns the value or default" pop semantics is documented for.
pub(crate) fn map_method(name: &str, _arity: usize, want_raw_option: bool) -> (String, Option<&'static str>) {
    match name {
        // len() returns usize; Boring's length/count returns int (isize).
        "length" | "count" => ("len".into(), Some(" as isize")),
        // len() called directly (not via length/count) — cast to usize so comparisons
        // with Boring's `uint` (usize) variables don't cause type mismatch errors.
        // (usize is len()'s native return type, so this is a no-op cast kept for symmetry.)
        "len"              => ("len".into(), Some(" as usize")),
        "isEmpty"          => ("is_empty".into(), None),
        "push"             => ("push".into(), None),
        // Vec::pop() returns Option<T>; unwrap to match Boring semantics (returns the value
        // or default) — UNLESS the caller wants the raw Option (see doc above).
        "pop" if want_raw_option => ("pop".into(), None),
        "pop"              => ("pop".into(), Some(".unwrap_or_default()")),
        "insert"           => ("insert".into(), None),
        "remove"           => ("remove".into(), None),
        "contains"         => ("contains".into(), None),
        "map"              => ("iter().cloned().map".into(), Some(".collect::<Vec<_>>()")),
        "filter"           => ("iter().cloned().filter".into(), Some(".collect::<Vec<_>>()")),
        // Collection search: find(closure) returns Option<T> (owned value, not a reference).
        "find"             => ("iter().cloned().find".into(), None),
        "indexOf"          => ("iter().position".into(), None),
        // position() on an iterator — use .cloned() so the closure receives owned T
        // values (not &T refs), keeping comparisons type-correct (kk == k).
        "position"         => ("cloned().position".into(), None),
        "reduce" | "fold"  => ("iter().cloned().fold".into(), None),
        "forEach" | "each" => ("iter().for_each".into(), None),
        "reversed"         => ("iter().rev().cloned().collect::<Vec<_>>".into(), None),
        // collect() — clone reference items so that iter-of-refs (e.g. keys())
        // gives owned Vec<T> instead of Vec<&T>, avoiding double-reference in closures.
        "collect"          => ("cloned().collect::<Vec<_>>".into(), None),
        "joined"           => ("join".into(), None),
        // split() returns an iterator in Rust; collect to Vec so .len() and indexing work.
        "split"            => ("split".into(), Some(".collect::<Vec<_>>()")),
        // chars() returns Chars iterator in Rust; collect to Vec<Arc<str>> so .len() and indexing work.
        "chars"            => ("chars().map(|c| Arc::<str>::from(c.to_string())).collect::<Vec<Arc<str>>>".into(), Some("")),
        "trim"             => ("trim".into(), None),
        "parse_int"        => ("parse::<isize>().ok".into(), Some("")),
        "parse_float"      => ("parse::<f64>().ok".into(), Some("")),
        "toUpperCase" | "uppercased" | "upper" | "to_upper" | "toUpper" => ("to_uppercase".into(), None),
        "toLowerCase" | "lowercased" | "lower" | "to_lower" | "toLower" => ("to_lowercase".into(), None),
        "startsWith" | "hasPrefix"   => ("starts_with".into(), None),
        "endsWith"   | "hasSuffix"   => ("ends_with".into(), None),
        // Vec::first()/last() return Option<&T> (a borrow); Boring's `first()`/`last()`
        // are documented (book.md's "Array methods" table) as returning an *owned* `T?`,
        // matching every other Vec-derived method here — append `.cloned()` so the
        // emitted Option actually holds T, not &T.
        "first"            => ("first".into(), Some(".cloned()")),
        "last"             => ("last".into(), Some(".cloned()")),
        // `arr.append(other)` merges a whole other collection in — always `.extend()`,
        // never `.push()` (that's `arr.push(v)`, a single element — see docs/book.md's
        // "Array methods" table). This was mapped to `push` here, which only "worked"
        // in the one call-site branch that separately detected a collection-typed arg
        // and overrode it back to `extend` (emit_methods.rs's `arg_is_collection`
        // check) — every other call site (the general fallback) had no such override
        // and emitted `arr.push(other_vec)`, a type mismatch (E0308: expected element
        // type, found `Vec<T>` — confirmed via examples/tokio.br's
        // `all_users.append(h_file.value)`).
        "append"           => ("extend".into(), None),
        "extend"           => ("extend".into(), None),
        // T'weak — .upgrade() returns Option<Rc/Arc<T>>; unwrap so the result is
        // the strong ref directly, matching the interpreter's semantics (upgrade returns
        // the object or nil). The panic message makes stale-ref bugs easier to diagnose.
        "upgrade"          => ("upgrade".into(), Some(".expect(\"attempted to use a stale weak reference\")")),
        // Collection index API — implemented by BoringArrayIndex / BoringDictIndex / BoringSetIndex
        // traits emitted in the file preamble.
        "firstIndex"       => ("first_index".into(), None),
        "nextIndex"        => ("next_index".into(), None),
        "removeAt"         => ("remove_at".into(), None),
        // get_at(i) — explicit positional read via opaque index (useful for sets where
        // `set[i]` is not valid Rust syntax for HashSet).
        "getAt"            => ("get_at".into(), None),
        // Fallback: convert any unrecognised camelCase method to snake_case so that
        // Boring callers can write e.g. `path.fileName()` and get `path.file_name()` in Rust.
        // (User-defined Boring struct methods are guarded before map_method is reached, so
        // they are unaffected by this conversion.)
        other              => (camel_to_snake(other), None),
    }
}


/// Map boring field names to Rust field names.
pub(crate) fn map_field(name: &str) -> &str {
    match name {
        // len() returns usize in Rust; Boring's `int` is isize — cast so the type matches.
        "length" | "count" => "len() as isize",
        "isEmpty" => "is_empty()",
        other => other,
    }
}

/// Map a Boring type name to its BoringError match arm(s): `(arm_pattern, error_rust_type,
/// error_binding_expr)`. String types produce two entries (Str for literals, String for
/// dynamic). Every non-scalar prim (String/Int/Float/Bool) keeps its established
/// `Arc<str>`-stringified binding, unchanged.
///
/// The twelve fixed-width kinds (int8..int128, uint8..uint128, float32, float64) are
/// different: they bind the *native* Rust type (`i8`, `u32`, `f32`, …), reconstructed
/// from `BoringError::Scalar`'s `(ScalarKind, u128)` payload (docs/float-width-types.md
/// §7) — real values, not stringified, since the whole point of `Scalar` over `Other`
/// is that the compiler already knows the exact concrete type at every one of these
/// call sites. This ALSO fixes a real bug, not just a missed optimization: every one of
/// these twelve type names used to fall into the generic `other` arm below, which
/// compiled to a **bare `ref __boring_other` binding pattern** — despite its
/// `/* unreachable catch {ty} */` comment, that pattern matches *any* `BoringError`
/// value unconditionally, so `catch Int8:` was silently catching every thrown error of
/// every type, not just `Int8`.
pub(crate) fn boring_type_to_boring_val_arms(ty: &str) -> Vec<(String, String, String)> {
    macro_rules! scalar_arm {
        ($kind:literal, $rust_ty:literal, $reinterp:literal) => {
            vec![(
                format!("BoringError::Scalar(ScalarKind::{}, __boring_bits)", $kind),
                $rust_ty.to_string(),
                format!("__boring_bits {}", $reinterp),
            )]
        };
    }
    match ty {
        "String" | "string" | "cstring" | "tstring" => vec![
            // &'static str literal
            ("BoringError::Str(__boring_s)".to_string(), "Arc<str>".to_string(),
             "Arc::<str>::from(__boring_s.to_string())".to_string()),
            // Arc<str> from interpolation or re-throw
            ("BoringError::String(ref __boring_s)".to_string(), "Arc<str>".to_string(),
             "__boring_s.clone()".to_string()),
        ],
        "Int" | "int" => vec![
            ("BoringError::Int(__boring_n)".to_string(), "Arc<str>".to_string(),
             "Arc::<str>::from(__boring_n.to_string())".to_string()),
        ],
        // `Float64`/`float64` are accepted spellings of `Float`/`float` here too —
        // both route through the same pre-existing `BoringError::Float` fast path,
        // not through `Scalar` (see `scalar_ctor_name`'s doc comment for why
        // float64 is deliberately excluded from that mechanism).
        "Float" | "float" | "Float64" | "float64" | "f64" => vec![
            ("BoringError::Float(__boring_f)".to_string(), "Arc<str>".to_string(),
             "Arc::<str>::from(__boring_f.to_string())".to_string()),
        ],
        "Bool" | "bool" => vec![
            ("BoringError::Bool(__boring_b)".to_string(), "Arc<str>".to_string(),
             "Arc::<str>::from(__boring_b.to_string())".to_string()),
        ],
        "Int8" | "int8"     => scalar_arm!("Int8", "i8", "as i128 as i8"),
        "Int16" | "int16"   => scalar_arm!("Int16", "i16", "as i128 as i16"),
        "Int32" | "int32"   => scalar_arm!("Int32", "i32", "as i128 as i32"),
        "Int64" | "int64"   => scalar_arm!("Int64", "i64", "as i128 as i64"),
        "Int128" | "int128" => scalar_arm!("Int128", "i128", "as i128"),
        "Uint8" | "uint8"     => scalar_arm!("Uint8", "u8", "as u8"),
        "Uint16" | "uint16"   => scalar_arm!("Uint16", "u16", "as u16"),
        "Uint32" | "uint32"   => scalar_arm!("Uint32", "u32", "as u32"),
        "Uint64" | "uint64"   => scalar_arm!("Uint64", "u64", "as u64"),
        "Uint128" | "uint128" => vec![(
            "BoringError::Scalar(ScalarKind::Uint128, __boring_bits)".to_string(),
            "u128".to_string(),
            "__boring_bits".to_string(),
        )],
        "Float32" | "float32" | "f32" => vec![(
            "BoringError::Scalar(ScalarKind::Float32, __boring_bits)".to_string(),
            "f32".to_string(),
            "f32::from_bits(__boring_bits as u32)".to_string(),
        )],
        other => vec![
            // Unknown type: will be handled by the named-clause path (BoringError::Other)
            (format!("/* unreachable catch {} */ ref __boring_other", other),
             "Arc<str>".to_string(),
             "Arc::<str>::from(__boring_other.to_string())".to_string()),
        ],
    }
}

pub(crate) fn escape_str(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '"'  => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0' => out.push_str("\\0"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c    => out.push(c),
        }
    }
    out
}

/// Like escape_str but does NOT escape `{` and `}`, so they pass through
/// as Rust format-string placeholders in println!/format! macro args.
pub(crate) fn escape_str_macro(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '"'  => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0' => out.push_str("\\0"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c    => out.push(c),
        }
    }
    out
}


/// Returns true if `expr` evaluates to a `std::time::Instant`.
///
/// Used to choose between `tokio::time::sleep` / `tokio::time::timeout`
/// (Duration-based) and `tokio::time::sleep_until` / `tokio::time::timeout_at`
/// (Instant-based).
///
/// Detects:
///   • `Instant.now()`                      — static call on the Instant type
///   • `Instant.now() + Duration.fromSecs(n)` — BinOp with an Instant on either side
///   • `deadline` where deadline ∈ instant_vars
pub(crate) fn expr_is_instant(expr: &Expr, instant_vars: &std::collections::HashSet<String>) -> bool {
    match &expr.kind {
        ExprKind::Var(name) => instant_vars.contains(name.as_str()),
        ExprKind::MethodCall(obj, _, _) | ExprKind::Call(obj, _) => {
            if let ExprKind::Var(type_name) = &obj.kind {
                if type_name.as_str() == "Instant" { return true; }
            }
            expr_is_instant(obj, instant_vars)
        }
        ExprKind::BinOp(_, lhs, rhs) => {
            expr_is_instant(lhs, instant_vars) || expr_is_instant(rhs, instant_vars)
        }
        _ => false,
    }
}

/// Convert a camelCase identifier to snake_case.
///
/// Used to let Boring callers write `Duration.fromSecs(5)` while the
/// generated Rust gets the idiomatic `Duration::from_secs(5)`.
///
/// Rules:
///   - An uppercase letter that follows a lowercase letter gets `_` prepended.
///   - Consecutive uppercase letters (acronyms like "URL", "HTTP") are kept
///     together with only one `_` before the run.
///
/// Examples:
///   fromSecs      → from_secs
///   fromMillis    → from_millis
///   fileName      → file_name
///   getHTTPClient → get_http_client  (run of uppercase treated as one word)
pub(crate) fn camel_to_snake(s: &str) -> String {
    if !s.chars().any(|c| c.is_uppercase()) {
        return s.to_string(); // already snake_case — fast path
    }
    let mut out = String::with_capacity(s.len() + 4);
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_uppercase() {
            let prev_lower = i > 0 && chars[i - 1].is_lowercase();
            let next_lower = i + 1 < n && chars[i + 1].is_lowercase();
            // Insert `_` before an uppercase letter when:
            //   • it follows a lowercase letter (camelCase boundary), OR
            //   • it's the start of a word within an all-caps run (e.g. "HTTPClient" → "http_client")
            if prev_lower || (i > 0 && next_lower && !out.ends_with('_')) {
                out.push('_');
            }
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Wrap a boring identifier in `r#` if it's a Rust keyword.
pub(crate) fn escape_rust_keyword(name: &str) -> String {
    match name {
        "fn" | "type" | "let" | "use" | "mod" | "impl" | "trait" | "enum" | "struct"
        | "match" | "loop" | "while" | "for" | "if" | "else" | "return" | "break"
        | "continue" | "move" | "ref" | "in" | "as" | "where" | "pub" | "super"
        | "crate" | "const" | "static" | "mut" | "unsafe" | "extern" | "async"
        | "await" | "dyn" | "box" | "abstract" | "become" | "do" | "final"
        | "override" | "priv" | "typeof" | "unsized" | "virtual" | "yield" | "try"
        => format!("r#{}", name),
        other => other.to_string(),
    }
}

/// Collect all variable names referenced in an expression (shallow — does not recurse
/// into nested closures, which have their own capture scope).
pub(crate) fn collect_var_names(expr: &Expr) -> Vec<String> {
    let mut out = Vec::new();
    collect_vars_in(expr, &mut out);
    out.sort();
    out.dedup();
    out
}

/// Collects every name a match-arm pattern binds (`Pattern::Bind`, recursively through
/// `Variant`/`Tuple`/`Some` sub-patterns). Used by `collect_vars_in`/`collect_vars_in_stmt`'s
/// `Match` handling to keep a pattern-bound name (e.g. `MInner(v)`'s `v`) from being reported
/// as a reference to some unrelated outer `v` of the same name — without this, a top-level
/// scripting `let v = ...` and an unrelated `match ...: Variant(v): v` elsewhere in the same
/// file collide: the arm's own tail-expression read of its bound `v` looks, to a naive
/// free-variable scan, identical to a genuine reference to the top-level `v`.
pub(crate) fn collect_pattern_bind_names(pats: &[Pattern], out: &mut std::collections::HashSet<String>) {
    for p in pats {
        match p {
            Pattern::Bind(name) => { out.insert(name.clone()); }
            Pattern::Variant(_, sub) | Pattern::Tuple(sub) => collect_pattern_bind_names(sub, out),
            Pattern::Some(inner) => collect_pattern_bind_names(std::slice::from_ref(inner.as_ref()), out),
            Pattern::Wildcard | Pattern::Lit(_) | Pattern::None => {}
        }
    }
}

pub(crate) fn collect_vars_in(expr: &Expr, out: &mut Vec<String>) {
    match &expr.kind {
        ExprKind::Var(name)                 => out.push(name.clone()),
        ExprKind::BinOp(_, l, r)            => { collect_vars_in(l, out); collect_vars_in(r, out); }
        ExprKind::UnaryOp(_, e)             => collect_vars_in(e, out),
        ExprKind::Field(e, _) | ExprKind::OptionalField(e, _) => collect_vars_in(e, out),
        ExprKind::Index(e, i)               => { collect_vars_in(e, out); collect_vars_in(i, out); }
        ExprKind::LabeledIndex(e, args)      => {
            collect_vars_in(e, out);
            for a in args { collect_vars_in(&a.value, out); }
        }
        ExprKind::Call(f, args) | ExprKind::MethodCall(f, _, args) | ExprKind::OptionalMethodCall(f, _, args) => {
            collect_vars_in(f, out);
            for a in args { collect_vars_in(&a.value, out); }
        }
        ExprKind::Pipe(lhs, _, args) => {
            collect_vars_in(lhs, out);
            for a in args { collect_vars_in(&a.value, out); }
        }
        ExprKind::Array(elems) | ExprKind::Tuple(elems) | ExprKind::Set(elems) => {
            for e in elems { collect_vars_in(e, out); }
        }
        ExprKind::ArrayFill { value, count } => {
            collect_vars_in(value, out); collect_vars_in(count, out);
        }
        ExprKind::ArrayAlloc { count } => { collect_vars_in(count, out); }
        ExprKind::ArrayComp { expr, count, .. } => {
            collect_vars_in(expr, out); collect_vars_in(count, out);
        }
        ExprKind::ArrayCompIter { expr, iter, .. } => {
            collect_vars_in(expr, out); collect_vars_in(iter, out);
        }
        ExprKind::LabeledArrayComp { expr, clauses } => {
            for (_, count) in clauses { collect_vars_in(count, out); }
            collect_vars_in(expr, out);
        }
        ExprKind::RelabelCast(e, _) => collect_vars_in(e, out),
        ExprKind::Dict(pairs) => {
            for (k, v) in pairs { collect_vars_in(k, out); collect_vars_in(v, out); }
        }
        ExprKind::Else(e, d) | ExprKind::TryElse(e, d) => {
            collect_vars_in(e, out); collect_vars_in(d, out);
        }
        ExprKind::TryElseBlock(try_stmts, else_stmts) => {
            for s in try_stmts { collect_vars_in_stmt(s, out); }
            for s in else_stmts { collect_vars_in_stmt(s, out); }
        }
        ExprKind::Cast(e, _)  => collect_vars_in(e, out),
        ExprKind::Assign(target, value) => { collect_vars_in(target, out); collect_vars_in(value, out); }
        ExprKind::StringInterp(segs) => {
            for seg in segs {
                match seg {
                    StringSegment::Expr(e) | StringSegment::FormattedExpr(e, _) => collect_vars_in(e, out),
                    StringSegment::Lit(_) => {}
                }
            }
        }
        ExprKind::Block(stmts) | ExprKind::Do(stmts) => {
            for s in stmts { collect_vars_in_stmt(s, out); }
        }
        ExprKind::Loop(s) => {
            for st in &s.body { collect_vars_in_stmt(st, out); }
        }
        ExprKind::JoinAll(exprs) => {
            for e in exprs { collect_vars_in(e, out); }
        }
        ExprKind::TaskWithTimeout(dur, body) => {
            collect_vars_in(dur, out);
            collect_vars_in(body, out);
        }

        // ── Previously missing — produced silent use-after-move in task bodies ──

        // `f<T>(args)` — generic call; type args carry no var refs
        ExprKind::GenericCall(callee, _type_args, args) => {
            collect_vars_in(callee, out);
            for a in args { collect_vars_in(&a.value, out); }
        }

        // Range literal `a..b` / `a..=b`
        ExprKind::Range { start, end, .. } => {
            collect_vars_in(start, out);
            collect_vars_in(end, out);
        }
        ExprKind::SliceRange { start, end, .. } => {
            if let Some(s) = start { collect_vars_in(s, out); }
            if let Some(e) = end   { collect_vars_in(e, out); }
        }

        // Closure: walk param defaults and body.
        // We intentionally do NOT recurse into the params' names — those introduce new
        // bindings rather than referencing outer variables. Defaults *are* evaluated in
        // the outer scope, so they can reference Arc vars that need cloning.
        ExprKind::Closure(params, _ret, body, _, _) => {
            for p in params {
                if let Some(default) = &p.default {
                    collect_vars_in(default, out);
                }
            }
            match body {
                ClosureBody::Expr(e) => collect_vars_in(e, out),
                ClosureBody::Block(stmts) => {
                    for s in stmts { collect_vars_in_stmt(s, out); }
                }
            }
        }

        // if/elif/else expression — walk all branch conditions and bodies
        ExprKind::If(if_stmt) => {
            for (cond, body) in &if_stmt.branches {
                collect_vars_in(cond, out);
                for s in body { collect_vars_in_stmt(s, out); }
            }
            if let Some(else_body) = &if_stmt.else_body {
                for s in else_body { collect_vars_in_stmt(s, out); }
            }
        }

        // match expression — walk subject and each arm (guard + body). Each arm's own
        // pattern-bound names (see `collect_pattern_bind_names`) are excluded from what
        // that arm's guard/body contribute -- they're local to the arm, not references
        // to an outer variable of the same name.
        ExprKind::Match(match_stmt) => {
            collect_vars_in(&match_stmt.subject, out);
            for arm in &match_stmt.arms {
                let mut bound = std::collections::HashSet::new();
                collect_pattern_bind_names(&arm.patterns, &mut bound);
                let mut arm_vars: Vec<String> = Vec::new();
                if let Some(guard) = &arm.guard { collect_vars_in(guard, &mut arm_vars); }
                match &arm.body {
                    MatchBody::Expr(e) => collect_vars_in(e, &mut arm_vars),
                    MatchBody::Block(stmts) => {
                        for s in stmts { collect_vars_in_stmt(s, &mut arm_vars); }
                    }
                }
                out.extend(arm_vars.into_iter().filter(|v| !bound.contains(v)));
            }
        }

        // task expression — walk the spawned body
        ExprKind::Task(inner) => collect_vars_in(inner, out),

        // Rust macro call — walk all argument expressions
        ExprKind::MacroCall { args, .. } => {
            for e in args { collect_vars_in(e, out); }
        }

        // Write-once / nil-coalescing assign: recurse both sides
        ExprKind::QuestionAssign(target, rhs) => { collect_vars_in(target, out); collect_vars_in(rhs, out); }

        ExprKind::New { arena, ctor } => {
            if let Some(a) = arena { collect_vars_in(a, out); }
            collect_vars_in(ctor, out);
        }

        ExprKind::KernelLaunch { config, kernel } => {
            if let Some(e) = &config.block { collect_vars_in(e, out); }
            if let Some(e) = &config.grid  { collect_vars_in(e, out); }
            if let Some(e) = &config.after { collect_vars_in(e, out); }
            collect_vars_in(kernel, out);
        }

        // Leaf nodes (no sub-expressions containing variable references)
        ExprKind::Int(_) | ExprKind::UInt64(_) | ExprKind::Float(_) | ExprKind::Str(_) | ExprKind::Bool(_)
        | ExprKind::Nil | ExprKind::Void | ExprKind::DotIdent(_) => {}
    }
}

pub(crate) fn collect_vars_in_stmt(stmt: &Stmt, out: &mut Vec<String>) {
    match stmt {
        Stmt::Expr(e) | Stmt::Return(ReturnStmt { value: Some(e), .. })
            | Stmt::Throw(ThrowStmt { value: Some(e), .. }) => collect_vars_in(e, out),
        Stmt::Let(l) => { if let Some(v) = &l.value { collect_vars_in(v, out); } }
        Stmt::If(i) => {
            for (cond, body) in &i.branches {
                collect_vars_in(cond, out);
                for s in body { collect_vars_in_stmt(s, out); }
            }
            if let Some(b) = &i.else_body { for s in b { collect_vars_in_stmt(s, out); } }
        }
        Stmt::While(w) => {
            collect_vars_in(&w.condition, out);
            for s in &w.body { collect_vars_in_stmt(s, out); }
        }
        Stmt::For(f) => {
            collect_vars_in(&f.iterable, out);
            for s in &f.body { collect_vars_in_stmt(s, out); }
        }
        Stmt::WhileLet(w) => {
            collect_vars_in(&w.value, out);
            for s in &w.body { collect_vars_in_stmt(s, out); }
        }
        Stmt::Try(t) => {
            for s in &t.body { collect_vars_in_stmt(s, out); }
            for clause in &t.catch_clauses {
                for s in &clause.body { collect_vars_in_stmt(s, out); }
            }
        }
        Stmt::Defer(body) => {
            for s in body { collect_vars_in_stmt(s, out); }
        }
        Stmt::Guard(g) => {
            match &g.cond {
                crate::ast::GuardCond::Expr(e) => collect_vars_in(e, out),
                crate::ast::GuardCond::Clauses(clauses) => {
                    for clause in clauses {
                        match clause {
                            crate::ast::CondClause::Expr(e) => collect_vars_in(e, out),
                            crate::ast::CondClause::Let(_, val) | crate::ast::CondClause::LetPat(_, val) => collect_vars_in(val, out),
                        }
                    }
                }
            }
            for s in &g.else_body { collect_vars_in_stmt(s, out); }
        }
        Stmt::Match(m) => {
            collect_vars_in(&m.subject, out);
            for arm in &m.arms {
                // Same arm-local pattern-bind exclusion as `collect_vars_in`'s
                // `ExprKind::Match` case — see `collect_pattern_bind_names`'s doc comment.
                let mut bound = std::collections::HashSet::new();
                collect_pattern_bind_names(&arm.patterns, &mut bound);
                let mut arm_vars: Vec<String> = Vec::new();
                if let Some(guard) = &arm.guard { collect_vars_in(guard, &mut arm_vars); }
                match &arm.body {
                    MatchBody::Expr(e) => collect_vars_in(e, &mut arm_vars),
                    MatchBody::Block(stmts) => {
                        for s in stmts { collect_vars_in_stmt(s, &mut arm_vars); }
                    }
                }
                out.extend(arm_vars.into_iter().filter(|v| !bound.contains(v)));
            }
        }
        _ => {}
    }
}

/// Collect the names of all variables *declared* (via `let`/`var`) inside `stmts`.
/// Used by the global-var promotion pass to exclude local re-declarations so that
/// a function with `var i = 0` inside its body doesn't incorrectly cause the top-level
/// `var i` to be promoted to a module-level static.
/// Returns `true` if `stmts` contains any statement that constitutes an early exit
/// (explicit `return`, `throw`, or `guard`).  Used by `emit_body` to decide whether
/// the `__deferred_ret` closure wrapper is actually needed.
pub(crate) fn body_has_early_return(stmts: &[Stmt]) -> bool {
    for stmt in stmts {
        match stmt {
            Stmt::Return(_) | Stmt::Throw(_) | Stmt::Guard(_) => return true,
            Stmt::If(i)
                if (i.branches.iter().any(|(_, body)| body_has_early_return(body))
                    || i.else_body.as_deref().is_some_and(body_has_early_return))
                => {
                    return true;
                }
            Stmt::While(w) if body_has_early_return(&w.body) => { return true; }
            Stmt::For(f) if body_has_early_return(&f.body) => { return true; }
            Stmt::WhileLet(w) if body_has_early_return(&w.body) => { return true; }
            Stmt::Try(t) => {
                if body_has_early_return(&t.body) { return true; }
                if t.catch_clauses.iter().any(|c| body_has_early_return(&c.body)) { return true; }
            }
            _ => {}
        }
    }
    false
}

pub(crate) fn collect_local_decl_names(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
    for stmt in stmts {
        match stmt {
            Stmt::Let(l) => { out.insert(l.name.clone()); }
            Stmt::If(i) => {
                for (_, body) in &i.branches { collect_local_decl_names(body, out); }
                if let Some(b) = &i.else_body { collect_local_decl_names(b, out); }
            }
            Stmt::While(w) => collect_local_decl_names(&w.body, out),
            Stmt::For(f) => {
                for v in &f.vars { out.insert(v.clone()); }
                collect_local_decl_names(&f.body, out);
            }
            Stmt::WhileLet(w) => {
                out.insert(w.name.clone());
                collect_local_decl_names(&w.body, out);
            }
            Stmt::Try(t) => {
                collect_local_decl_names(&t.body, out);
                for c in &t.catch_clauses { collect_local_decl_names(&c.body, out); }
            }
            _ => {}
        }
    }
}

/// Walks the whole program looking for struct-construction calls that use the
/// `_` fill-rest marker (`Arg::default_rest`, see `src/ast/mod.rs`) and records
/// the callee name into `out`. Consulted by `emit_struct` (`emit_struct.rs`) to
/// conditionally add `Default` to the struct's derive list: `_` lowers to a
/// trailing `..Default::default()` (`emit_constructor_inner`,
/// `emit_expr.rs`), which requires the target type to implement `Default`.
/// Only matters for Boring-*owned* structs — the motivating external case
/// (e.g. Bevy's `Transform`) already implements `Default` in its own crate
/// and needs no derive from us; those calls just never match any name here.
///
/// Best-effort, not exhaustively complete over every `ExprKind`: an unlisted
/// nesting form degrades to "the struct doesn't get `#[derive(Default)]`",
/// which surfaces at `cargo build` time as rustc's own
/// "the trait `Default` is not implemented" diagnostic — a safe, if less
/// friendly, failure mode rather than a silent miscompile.
pub(crate) fn collect_default_rest_targets(items: &[Item], out: &mut std::collections::HashSet<String>) {
    for item in items {
        match item {
            Item::Fn(f) => scan_stmts_for_default_rest(&f.body, out),
            Item::Struct(s) => {
                for m in &s.methods { scan_stmts_for_default_rest(&m.body, out); }
                for i in &s.inits { scan_stmts_for_default_rest(&i.body, out); }
                for tm in &s.type_methods { scan_stmts_for_default_rest(&tm.body, out); }
                for st in &s.setters { scan_stmts_for_default_rest(&st.body, out); }
                for f in &s.fields {
                    if let Some(d) = &f.default { scan_expr_for_default_rest(d, out); }
                }
            }
            Item::Enum(e) => {
                for m in &e.methods { scan_stmts_for_default_rest(&m.body, out); }
                for st in &e.setters { scan_stmts_for_default_rest(&st.body, out); }
                for tm in &e.type_methods { scan_stmts_for_default_rest(&tm.body, out); }
            }
            Item::Ext(e) => {
                for m in &e.methods { scan_stmts_for_default_rest(&m.body, out); }
                for st in &e.setters { scan_stmts_for_default_rest(&st.body, out); }
            }
            Item::Trait(t) => {
                for m in &t.defaults { scan_stmts_for_default_rest(&m.body, out); }
            }
            Item::Let(s) => {
                if let Some(v) = &s.value { scan_expr_for_default_rest(v, out); }
            }
            Item::Mod(m) => collect_default_rest_targets(&m.items, out),
            Item::Stmt(s) => scan_stmt_for_default_rest(s, out),
            _ => {}
        }
    }
}

fn scan_stmts_for_default_rest(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
    for s in stmts { scan_stmt_for_default_rest(s, out); }
}

fn scan_cond_clause_for_default_rest(c: &CondClause, out: &mut std::collections::HashSet<String>) {
    match c {
        CondClause::Let(_, e) | CondClause::LetPat(_, e) | CondClause::Expr(e) => scan_expr_for_default_rest(e, out),
    }
}

fn scan_stmt_for_default_rest(stmt: &Stmt, out: &mut std::collections::HashSet<String>) {
    match stmt {
        Stmt::Let(l) => { if let Some(v) = &l.value { scan_expr_for_default_rest(v, out); } }
        Stmt::LetDestructure(l) => scan_expr_for_default_rest(&l.value, out),
        Stmt::Return(r) => { if let Some(v) = &r.value { scan_expr_for_default_rest(v, out); } }
        Stmt::Throw(t) => { if let Some(v) = &t.value { scan_expr_for_default_rest(v, out); } }
        Stmt::If(i) => {
            for (cond, body) in &i.branches {
                scan_expr_for_default_rest(cond, out);
                scan_stmts_for_default_rest(body, out);
            }
            if let Some(b) = &i.else_body { scan_stmts_for_default_rest(b, out); }
        }
        Stmt::IfLet(i) => {
            for c in &i.clauses { scan_cond_clause_for_default_rest(c, out); }
            scan_stmts_for_default_rest(&i.then_body, out);
            for br in &i.elif_branches {
                for c in &br.clauses { scan_cond_clause_for_default_rest(c, out); }
                scan_stmts_for_default_rest(&br.body, out);
            }
            if let Some(b) = &i.else_body { scan_stmts_for_default_rest(b, out); }
        }
        Stmt::Match(m) => {
            scan_expr_for_default_rest(&m.subject, out);
            for arm in &m.arms {
                if let Some(g) = &arm.guard { scan_expr_for_default_rest(g, out); }
                match &arm.body {
                    MatchBody::Expr(e) => scan_expr_for_default_rest(e, out),
                    MatchBody::Block(b) => scan_stmts_for_default_rest(b, out),
                }
            }
        }
        Stmt::While(w) => { scan_expr_for_default_rest(&w.condition, out); scan_stmts_for_default_rest(&w.body, out); }
        Stmt::WhileLet(w) => { scan_expr_for_default_rest(&w.value, out); scan_stmts_for_default_rest(&w.body, out); }
        Stmt::DoWhile(d) => { scan_stmts_for_default_rest(&d.body, out); scan_expr_for_default_rest(&d.condition, out); }
        Stmt::Loop(l) => scan_stmts_for_default_rest(&l.body, out),
        Stmt::Wait(e, _) => scan_expr_for_default_rest(e, out),
        Stmt::For(f) => { scan_expr_for_default_rest(&f.iterable, out); scan_stmts_for_default_rest(&f.body, out); }
        Stmt::Guard(g) => {
            match &g.cond {
                GuardCond::Expr(e) => scan_expr_for_default_rest(e, out),
                GuardCond::Clauses(cs) => for c in cs { scan_cond_clause_for_default_rest(c, out); },
            }
            scan_stmts_for_default_rest(&g.else_body, out);
        }
        Stmt::Try(t) => {
            scan_stmts_for_default_rest(&t.body, out);
            for c in &t.catch_clauses { scan_stmts_for_default_rest(&c.body, out); }
        }
        Stmt::Defer(b) => scan_stmts_for_default_rest(b, out),
        Stmt::Expr(e) => scan_expr_for_default_rest(e, out),
        Stmt::Fn(f) => scan_stmts_for_default_rest(&f.body, out),
        Stmt::Struct(s) => {
            for m in &s.methods { scan_stmts_for_default_rest(&m.body, out); }
            for i in &s.inits { scan_stmts_for_default_rest(&i.body, out); }
        }
        Stmt::Enum(e) => { for m in &e.methods { scan_stmts_for_default_rest(&m.body, out); } }
        Stmt::Mod(m) => collect_default_rest_targets(&m.items, out),
        Stmt::Yield(e, _) => scan_expr_for_default_rest(e, out),
        Stmt::KernelBlock(k) => scan_stmts_for_default_rest(&k.body, out),
        Stmt::With(w) => scan_stmts_for_default_rest(&w.body, out),
        Stmt::Break(_, Some(e)) => scan_expr_for_default_rest(e, out),
        _ => {}
    }
}

fn scan_expr_for_default_rest(expr: &Expr, out: &mut std::collections::HashSet<String>) {
    match &expr.kind {
        ExprKind::Call(callee, args) => {
            scan_expr_for_default_rest(callee, out);
            if args.iter().any(|a| a.default_rest) {
                if let ExprKind::Var(name) = &callee.kind {
                    out.insert(name.clone());
                }
            }
            for a in args { scan_expr_for_default_rest(&a.value, out); }
        }
        ExprKind::MethodCall(recv, _, args) | ExprKind::OptionalMethodCall(recv, _, args) => {
            scan_expr_for_default_rest(recv, out);
            for a in args { scan_expr_for_default_rest(&a.value, out); }
        }
        ExprKind::GenericCall(callee, _, args) => {
            scan_expr_for_default_rest(callee, out);
            for a in args { scan_expr_for_default_rest(&a.value, out); }
        }
        ExprKind::Pipe(lhs, _, args) => {
            scan_expr_for_default_rest(lhs, out);
            for a in args { scan_expr_for_default_rest(&a.value, out); }
        }
        ExprKind::BinOp(_, l, r) | ExprKind::Assign(l, r) | ExprKind::QuestionAssign(l, r)
            | ExprKind::Else(l, r) | ExprKind::TryElse(l, r) => {
            scan_expr_for_default_rest(l, out);
            scan_expr_for_default_rest(r, out);
        }
        ExprKind::UnaryOp(_, e) | ExprKind::Field(e, _) | ExprKind::OptionalField(e, _)
            | ExprKind::Cast(e, _) => scan_expr_for_default_rest(e, out),
        ExprKind::Index(obj, idx) => { scan_expr_for_default_rest(obj, out); scan_expr_for_default_rest(idx, out); }
        ExprKind::LabeledIndex(obj, args) => {
            scan_expr_for_default_rest(obj, out);
            for a in args { scan_expr_for_default_rest(&a.value, out); }
        }
        ExprKind::New { arena, ctor } => {
            if let Some(a) = arena { scan_expr_for_default_rest(a, out); }
            scan_expr_for_default_rest(ctor, out);
        }
        ExprKind::TryElseBlock(body, else_body) => {
            scan_stmts_for_default_rest(body, out);
            scan_stmts_for_default_rest(else_body, out);
        }
        ExprKind::Array(elems) | ExprKind::Tuple(elems) | ExprKind::Set(elems) | ExprKind::JoinAll(elems) => {
            for e in elems { scan_expr_for_default_rest(e, out); }
        }
        ExprKind::ArrayFill { value, count } => {
            scan_expr_for_default_rest(value, out);
            scan_expr_for_default_rest(count, out);
        }
        ExprKind::ArrayAlloc { count } => scan_expr_for_default_rest(count, out),
        ExprKind::ArrayComp { expr, count, .. } => {
            scan_expr_for_default_rest(expr, out);
            scan_expr_for_default_rest(count, out);
        }
        ExprKind::ArrayCompIter { expr, iter, .. } => {
            scan_expr_for_default_rest(expr, out);
            scan_expr_for_default_rest(iter, out);
        }
        ExprKind::LabeledArrayComp { expr, clauses } => {
            scan_expr_for_default_rest(expr, out);
            for (_, c) in clauses { scan_expr_for_default_rest(c, out); }
        }
        ExprKind::Dict(pairs) => {
            for (k, v) in pairs {
                scan_expr_for_default_rest(k, out);
                scan_expr_for_default_rest(v, out);
            }
        }
        ExprKind::Range { start, end, .. } => {
            scan_expr_for_default_rest(start, out);
            scan_expr_for_default_rest(end, out);
        }
        ExprKind::SliceRange { start, end, .. } => {
            if let Some(s) = start { scan_expr_for_default_rest(s, out); }
            if let Some(e) = end { scan_expr_for_default_rest(e, out); }
        }
        ExprKind::RelabelCast(e, _) => scan_expr_for_default_rest(e, out),
        ExprKind::Closure(_, _, body, _, _) => match body {
            ClosureBody::Expr(e) => scan_expr_for_default_rest(e, out),
            ClosureBody::Block(b) => scan_stmts_for_default_rest(b, out),
        },
        ExprKind::If(i) => {
            for (cond, body) in &i.branches {
                scan_expr_for_default_rest(cond, out);
                scan_stmts_for_default_rest(body, out);
            }
            if let Some(b) = &i.else_body { scan_stmts_for_default_rest(b, out); }
        }
        ExprKind::Match(m) => {
            scan_expr_for_default_rest(&m.subject, out);
            for arm in &m.arms {
                if let Some(g) = &arm.guard { scan_expr_for_default_rest(g, out); }
                match &arm.body {
                    MatchBody::Expr(e) => scan_expr_for_default_rest(e, out),
                    MatchBody::Block(b) => scan_stmts_for_default_rest(b, out),
                }
            }
        }
        ExprKind::Block(b) | ExprKind::Do(b) => scan_stmts_for_default_rest(b, out),
        ExprKind::Loop(l) => scan_stmts_for_default_rest(&l.body, out),
        ExprKind::Task(e) => scan_expr_for_default_rest(e, out),
        ExprKind::TaskWithTimeout(a, b) => {
            scan_expr_for_default_rest(a, out);
            scan_expr_for_default_rest(b, out);
        }
        ExprKind::StringInterp(segs) => {
            for seg in segs {
                match seg {
                    StringSegment::Expr(e) | StringSegment::FormattedExpr(e, _) => scan_expr_for_default_rest(e, out),
                    StringSegment::Lit(_) => {}
                }
            }
        }
        ExprKind::MacroCall { args, .. } => { for a in args { scan_expr_for_default_rest(a, out); } }
        ExprKind::KernelLaunch { kernel, .. } => scan_expr_for_default_rest(kernel, out),
        _ => {}
    }
}

/// Records which of `candidates` (top-level immutable `let` scalar-constant names,
/// per `Transpiler::top_level_let_is_const_safe`) are referenced as free variables
/// inside `body` — a function or struct/enum/ext method body — and inserts each into
/// `out`. Mirrors the `Item::Fn`-only free-variable scan `pre_scan` already runs for
/// mutable top-level `var`s (see `global_vars_used_in_fns`), but is called for every
/// method body too (not just top-level functions), since a top-level `let` referenced
/// only from inside a struct/enum method — the shape the module-constant transpiler bug
/// this was written for actually reproduced with — is otherwise invisible to that scan.
pub(crate) fn collect_const_let_usage(
    candidates: &std::collections::HashSet<String>,
    params: &[Param],
    body: &[Stmt],
    out: &mut std::collections::HashSet<String>,
) {
    if candidates.is_empty() { return; }
    let param_names: std::collections::HashSet<String> = params.iter()
        .map(|p| p.name.clone()).collect();
    let mut local_decls: std::collections::HashSet<String> = std::collections::HashSet::new();
    collect_local_decl_names(body, &mut local_decls);
    let mut body_vars: Vec<String> = Vec::new();
    for stmt in body {
        collect_vars_in_stmt(stmt, &mut body_vars);
    }
    for v in &body_vars {
        if candidates.contains(v) && !param_names.contains(v) && !local_decls.contains(v) {
            out.insert(v.clone());
        }
    }
}

/// Returns `true` if any top-level item in the list contains a `task expr` (either as a
/// standalone detached task statement or as the RHS of a `let` binding).  Used by
/// `emit_program` to decide whether the auto-generated `main` needs to be `async`.
/// Returns true if any `for item in stream_fn():` appears in `stmts`
/// (direct call to a known stream function as the for-loop iterable).
pub(crate) fn body_has_stream_for(stmts: &[Stmt], stream_fns: &std::collections::HashSet<String>) -> bool {
    for stmt in stmts {
        match stmt {
            Stmt::For(f) => {
                if let ExprKind::Call(callee, _) = &f.iterable.kind {
                    if let ExprKind::Var(name) = &callee.kind {
                        if stream_fns.contains(name.as_str()) { return true; }
                    }
                }
                if body_has_stream_for(&f.body, stream_fns) { return true; }
            }
            Stmt::If(i)
                if (i.branches.iter().any(|(_, body)| body_has_stream_for(body, stream_fns))
                    || i.else_body.as_deref().is_some_and(|b| body_has_stream_for(b, stream_fns)))
                => { return true; }
            Stmt::While(w) if body_has_stream_for(&w.body, stream_fns)  => { return true; }
            Stmt::Defer(b) if body_has_stream_for(b,        stream_fns)  => { return true; }
            _ => {}
        }
    }
    false
}

/// Returns true if any `channel<T>(n)` call or `task:` expression appears in `stmts`.
/// Used to detect that `main` (or another function) needs to be async.
pub(crate) fn body_has_channel_or_task(stmts: &[Stmt]) -> bool {
    for stmt in stmts {
        match stmt {
            Stmt::Let(l)
                if l.value.as_ref().is_some_and(expr_has_channel_or_task) => { return true; }
            Stmt::Expr(e) | Stmt::Return(ReturnStmt { value: Some(e), .. })
                if expr_has_channel_or_task(e) => { return true; }
            Stmt::If(i)
                if (i.branches.iter().any(|(_, b)| body_has_channel_or_task(b))
                    || i.else_body.as_deref().is_some_and(body_has_channel_or_task))
                => { return true; }
            Stmt::While(w) if body_has_channel_or_task(&w.body) => { return true; }
            Stmt::For(f) if body_has_channel_or_task(&f.body) => { return true; }
            Stmt::Defer(b) if body_has_channel_or_task(b) => { return true; }
            _ => {}
        }
    }
    false
}

pub(crate) fn expr_has_channel_or_task(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::GenericCall(callee, _, _) => {
            matches!(&callee.kind, ExprKind::Var(n) if n == "channel")
        }
        ExprKind::Task(_) | ExprKind::TaskWithTimeout(..) => true,
        ExprKind::Block(stmts) => body_has_channel_or_task(stmts),
        _ => false,
    }
}

/// Returns true when a `task expr` expression should be spawned with
/// `tokio::task::spawn_blocking` instead of `tokio::spawn`.
///
/// Rules:
///   • `task syncFn(args)`    — syncFn is NOT in task_fns  → blocking
///   • `task: { sync block }` — block has no async content  → blocking
///   • Everything else is treated as async (conservative).
pub(crate) fn is_blocking_spawn(e: &Expr, task_fns: &std::collections::HashSet<String>) -> bool {
    match &e.kind {
        // Function call: blocking iff the callee is a known plain (non-task) function.
        // `task syncFn(args)` → spawn_blocking
        // `task asyncFn(args)` → tokio::spawn (asyncFn ∈ task_fns)
        ExprKind::Call(callee, _) => {
            if let ExprKind::Var(fn_name) = &callee.kind {
                !task_fns.contains(fn_name.as_str())
            } else {
                false // complex callee → conservative: async
            }
        }
        // Blocks: always async — blocks may contain channel sends, actor method calls,
        // or other async operations that are not visible without the transpiler's full
        // type-tracking state.  A future refinement could make this smarter by passing
        // the channel/actor variable sets; for now we default to safe (async).
        _ => false,
    }
}

/// Returns true when any statement in `stmts` calls a function from `task_fns`
/// (async functions like `wait`, `timeout`, or user-defined `task` functions).
/// Used to auto-promote `def main():` to async without requiring `task main():`.
pub(crate) fn body_calls_task_fn(stmts: &[Stmt], task_fns: &std::collections::HashSet<String>) -> bool {
    for stmt in stmts {
        match stmt {
            Stmt::Expr(e) | Stmt::Return(ReturnStmt { value: Some(e), .. })
                if expr_calls_task_fn(e, task_fns) => { return true; }
            Stmt::Let(l)
                if l.value.as_ref().is_some_and(|v| expr_calls_task_fn(v, task_fns)) => { return true; }
            Stmt::If(i)
                if (i.branches.iter().any(|(_, b)| body_calls_task_fn(b, task_fns))
                    || i.else_body.as_deref().is_some_and(|b| body_calls_task_fn(b, task_fns)))
                => { return true; }
            Stmt::While(w) if body_calls_task_fn(&w.body, task_fns) => { return true; }
            Stmt::For(f) if body_calls_task_fn(&f.body, task_fns) => { return true; }
            Stmt::Try(t)   => {
                if body_calls_task_fn(&t.body, task_fns) { return true; }
                if t.catch_clauses.iter().any(|c| body_calls_task_fn(&c.body, task_fns)) { return true; }
            }
            _ => {}
        }
    }
    false
}

/// Returns true when the stream body contains no async operations (no `wait`, no task fn calls,
/// no `task` expressions). Used to decide whether to emit an `Iterator` instead of an async stream.
pub(crate) fn body_is_sequential(stmts: &[Stmt], task_fns: &std::collections::HashSet<String>) -> bool {
    !body_has_wait(stmts) && !body_calls_task_fn(stmts, task_fns)
}

fn body_has_wait(stmts: &[Stmt]) -> bool {
    for stmt in stmts {
        match stmt {
            Stmt::Wait(..) => return true,
            Stmt::If(i)
                if (i.branches.iter().any(|(_, b)| body_has_wait(b))
                    || i.else_body.as_deref().is_some_and(body_has_wait))
                => { return true; }
            Stmt::While(w) if body_has_wait(&w.body) => { return true; }
            Stmt::For(f) if body_has_wait(&f.body) => { return true; }
            Stmt::Defer(b) if body_has_wait(b)       => { return true; }
            Stmt::Try(t)   => {
                if body_has_wait(&t.body) { return true; }
                if t.catch_clauses.iter().any(|c| body_has_wait(&c.body)) { return true; }
            }
            _ => {}
        }
    }
    false
}

fn expr_calls_task_fn(expr: &Expr, task_fns: &std::collections::HashSet<String>) -> bool {
    match &expr.kind {
        ExprKind::Call(callee, args) | ExprKind::MethodCall(callee, _, args) => {
            if let ExprKind::Var(name) = &callee.kind {
                if task_fns.contains(name.as_str()) { return true; }
            }
            args.iter().any(|a| expr_calls_task_fn(&a.value, task_fns))
                || expr_calls_task_fn(callee, task_fns)
        }
        ExprKind::Task(_) | ExprKind::TaskWithTimeout(..) => true,
        ExprKind::Block(stmts) => body_calls_task_fn(stmts, task_fns),
        ExprKind::BinOp(_, l, r) => {
            expr_calls_task_fn(l, task_fns) || expr_calls_task_fn(r, task_fns)
        }
        ExprKind::Closure(_, _, body, _, _) => match body {
            ClosureBody::Expr(e) => expr_calls_task_fn(e, task_fns),
            ClosureBody::Block(stmts) => body_calls_task_fn(stmts, task_fns),
        },
        _ => false,
    }
}

pub(crate) fn items_have_task(items: &[&Item]) -> bool {
    for item in items {
        match item {
            Item::Stmt(s) if stmt_has_task(s) => { return true; }
            Item::Let(l) if l.value.as_ref().is_some_and(expr_has_task) => { return true; }
            _ => {}
        }
    }
    false
}

pub(crate) fn items_have_task_call(items: &[&Item], task_fns: &std::collections::HashSet<String>) -> bool {
    for item in items {
        match item {
            Item::Stmt(Stmt::Expr(e)) if expr_has_task_call(e, task_fns) => { return true; }
            Item::Let(l) if l.value.as_ref().is_some_and(|v| expr_has_task_call(v, task_fns)) => { return true; }
            _ => {}
        }
    }
    false
}

pub(crate) fn stmt_has_task(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Expr(e) => expr_has_task(e),
        Stmt::Let(l)  => l.value.as_ref().is_some_and(expr_has_task),
        Stmt::If(i) => {
            i.branches.iter().any(|(cond, body)| {
                expr_has_task(cond) || body.iter().any(stmt_has_task)
            }) || i.else_body.as_deref().is_some_and(|b| b.iter().any(stmt_has_task))
        }
        Stmt::While(w) => expr_has_task(&w.condition) || w.body.iter().any(stmt_has_task),
        Stmt::For(f)   => expr_has_task(&f.iterable) || f.body.iter().any(stmt_has_task),
        Stmt::Defer(b) => b.iter().any(stmt_has_task),
        _ => false,
    }
}

pub(crate) fn expr_has_task(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Task(_) | ExprKind::TaskWithTimeout(..) => true,
        ExprKind::GenericCall(callee, _, _) =>
            matches!(&callee.kind, ExprKind::Var(n) if matches!(n.as_str(), "channel" | "oneshot" | "broadcast" | "watch")),
        ExprKind::Call(callee, _) =>
            matches!(&callee.kind, ExprKind::Var(n) if matches!(n.as_str(), "channel" | "oneshot" | "broadcast" | "watch")),
        ExprKind::Closure(_, _, _, _, task) if *task => true,
        _ => false,
    }
}

pub(crate) fn expr_has_task_call(expr: &Expr, task_fns: &std::collections::HashSet<String>) -> bool {
    match &expr.kind {
        ExprKind::Call(callee, args) => {
            if matches!(&callee.kind, ExprKind::Var(n) if task_fns.contains(n.as_str())) {
                return true;
            }
            // Recurse into arguments (e.g. `print runw(work)` nests task call inside print).
            args.iter().any(|a| expr_has_task_call(&a.value, task_fns))
        }
        _ => expr_has_task(expr),
    }
}

/// Map a boring trait name in a where-clause constraint to its Rust equivalent.
pub(crate) fn map_trait_bound(name: &str) -> String {
    match name {
        "Display"   => "std::fmt::Display".into(),
        "Debug"     => "std::fmt::Debug".into(),
        "Clone"     => "Clone".into(),
        "Copy"      => "Copy".into(),
        "PartialEq" => "PartialEq".into(),
        "Eq"        => "Eq".into(),
        "Hash"      => "std::hash::Hash".into(),
        "PartialOrd"=> "PartialOrd".into(),
        "Ord"       => "Ord".into(),
        "Default"   => "Default".into(),
        "Send"      => "Send".into(),
        "Sync"      => "Sync".into(),
        other       => other.into(),
    }
}

/// Emit a single generic parameter declaration.
/// `"$N:usize"` → `"const N: usize"`, lifetimes pass through, regular type params pass through.
pub(crate) fn emit_generic_param(p: &str) -> String {
    if let Some(rest) = p.strip_prefix('$') {
        if let Some((name, rust_ty)) = rest.split_once(':') {
            return format!("const {}: {}", name, rust_ty);
        }
    }
    p.to_string()
}

/// Extract the use-site name from a (possibly const-encoded) type parameter.
/// `"$N:usize"` → `"N"`, anything else passes through unchanged.
pub(crate) fn type_param_use_name(p: &str) -> String {
    if let Some(rest) = p.strip_prefix('$') {
        if let Some((name, _)) = rest.split_once(':') {
            return name.to_string();
        }
    }
    p.to_string()
}

pub(crate) fn type_params_str(params: &[String]) -> String {
    if params.is_empty() { String::new() }
    else {
        let parts: Vec<String> = params.iter().map(|p| emit_generic_param(p)).collect();
        format!("<{}>", parts.join(", "))
    }
}

/// Like `type_params_str` but adds `: Clone` bound to each regular type parameter.
/// Const generic params (`$N:usize`) are emitted as `const N: usize` without a Clone bound.
/// Used for `impl<T: Clone> Struct<T>` headers so that method bodies can call `.clone()`.
pub(crate) fn type_params_impl_str(params: &[String]) -> String {
    if params.is_empty() { String::new() }
    else {
        let bounded: Vec<String> = params.iter()
            .map(|p| {
                if p.starts_with('\'') { p.clone() }
                else if p.starts_with('$') { emit_generic_param(p) }
                else { format!("{}: Clone", p) }
            })
            .collect();
        format!("<{}>", bounded.join(", "))
    }
}

pub(crate) fn type_params_use_str(params: &[String]) -> String {
    if params.is_empty() { String::new() }
    else {
        let parts: Vec<String> = params.iter().map(|p| type_param_use_name(p)).collect();
        format!("<{}>", parts.join(", "))
    }
}

/// Collect all variant names from a pattern (non-recursive into nested struct patterns).
pub(crate) fn collect_pattern_variants(pat: &Pattern, out: &mut Vec<String>) {
    match pat {
        Pattern::Variant(name, _) => out.push(name.clone()),
        // Pattern::Some represents a `Some(...)` pattern — treat "Some" as a variant name
        // so that infer_match_enum can find which enum owns this variant.
        Pattern::Some(inner) => {
            out.push("Some".to_string());
            collect_pattern_variants(inner, out);
        }
        Pattern::None => out.push("None".to_string()),
        _ => {}
    }
}

/// Maps a resolved fixed-width numeric `Type` (int8..int128, uint8..uint128, float32,
/// float64 — the twelve `ScalarKind`/`BoringError::scalar_*` kinds, see
/// docs/float-width-types.md §7) to the name of its `BoringError::scalar_*`
/// constructor, for `throw`-value emission. `None` for the flexible `int`/`uint`/
/// `float` kinds (they already have their own `BoringError::Int`/`Float` fast path,
/// unaffected by this) and for every non-numeric type.
pub(crate) fn scalar_ctor_name(ty: &Type) -> Option<&'static str> {
    match ty {
        Type::Int8 => Some("scalar_i8"),
        Type::Int16 => Some("scalar_i16"),
        Type::Int32 => Some("scalar_i32"),
        Type::Int64 => Some("scalar_i64"),
        Type::Int128 => Some("scalar_i128"),
        Type::Uint8 => Some("scalar_u8"),
        Type::Uint16 => Some("scalar_u16"),
        Type::Uint32 => Some("scalar_u32"),
        Type::Uint64 => Some("scalar_u64"),
        Type::Uint128 => Some("scalar_u128"),
        // float64 deliberately excluded — `float`/`float64` already have their own
        // pre-existing fast path (`BoringError::Float(f64)`, covering the common
        // literal-throw case), unaffected by this feature. Routing it through
        // `Scalar` too would create two disjoint representations for the same
        // type depending on whether the thrown expression was a literal or a
        // variable, with no single `catch` clause spelling that reliably catches
        // both — see docs/float-width-types.md and the exec.rs `catch Float:` /
        // `catch Float64:` alias-matching note. `float32` has no such pre-existing
        // path (it was never a distinct type before), so it has no equivalent
        // conflict and stays routed through `Scalar`.
        Type::Float32 => Some("scalar_f32"),
        Type::Qualified(inner, _) => scalar_ctor_name(inner),
        Type::Named(n) => match n.as_str() {
            "int8" | "i8" => Some("scalar_i8"),
            "int16" | "i16" => Some("scalar_i16"),
            "int32" | "i32" => Some("scalar_i32"),
            "int64" | "i64" => Some("scalar_i64"),
            "int128" | "i128" => Some("scalar_i128"),
            "uint8" | "u8" => Some("scalar_u8"),
            "uint16" | "u16" => Some("scalar_u16"),
            "uint32" | "u32" => Some("scalar_u32"),
            "uint64" | "u64" => Some("scalar_u64"),
            "uint128" | "u128" => Some("scalar_u128"),
            "float32" | "f32" => Some("scalar_f32"),
            _ => None,
        },
        _ => None,
    }
}

/// Returns true if the Rust type string is a specific numeric type that may need coercion.
pub(crate) fn is_specific_numeric_type(ty: &str) -> bool {
    matches!(ty, "i8" | "i16" | "i32" | "i64" | "i128" | "isize"
               | "u8" | "u16" | "u32" | "u64" | "u128" | "usize"
               | "f32" | "f64")
}

/// Returns the wider of two numeric types (the one that can hold both values).
pub(crate) fn wider_numeric_type(a: &str, b: &str) -> String {
    // Rank: i8 < i16 < i32 < i64 < isize; u8 < u16 < u32 < u64 < usize; f32 < f64
    // Cross-family: prefer signed over unsigned, prefer float over int.
    fn rank(t: &str) -> (i32, i32) { // (family: 0=uint,1=int,2=float, size)
        match t {
            "u8"    => (0, 8),
            "u16"   => (0, 16),
            "u32"   => (0, 32),
            "u64"   => (0, 64),
            "u128"  => (0, 128),
            "usize" => (0, 64),
            "i8"    => (1, 8),
            "i16"   => (1, 16),
            "i32"   => (1, 32),
            "i64"   => (1, 64),
            "i128"  => (1, 128),
            "isize" => (1, 64),
            "f32"   => (2, 32),
            "f64"   => (2, 64),
            _       => (1, 64), // default to i64
        }
    }
    let (af, as_) = rank(a);
    let (bf, bs) = rank(b);
    // Pick the wider family first (float > int > uint), then wider size.
    if af > bf { return a.to_string(); }
    if bf > af { return b.to_string(); }
    // Same family: pick the wider one.
    if as_ >= bs { a.to_string() } else { b.to_string() }
}

/// Returns true if `stmts` contain a MatchStmt whose subject is a variable typed with a
/// type parameter (either the variable name IS a type param, or its declared type is), and
/// at least one arm pattern is a struct found in `struct_field_names`.
/// `type_param_var_names` is the set of variable names whose declared types are type params.
/// Used by `emit_fn` to detect generic-struct pattern matching and add `std::any::Any` bounds.
pub(crate) fn stmts_have_struct_match(
    stmts: &[Stmt],
    type_param_var_names: &std::collections::HashSet<String>,
    struct_field_names: &std::collections::HashSet<String>,
) -> bool {
    for stmt in stmts {
        match stmt {
            Stmt::Match(m) => {
                if let ExprKind::Var(vname) = &m.subject.kind {
                    if type_param_var_names.contains(vname.as_str()) {
                        let has_struct_arm = m.arms.iter().any(|arm| {
                            arm.patterns.iter().any(|p| {
                                if let Pattern::Variant(name, _) = p {
                                    struct_field_names.contains(name.as_str())
                                } else {
                                    false
                                }
                            })
                        });
                        if has_struct_arm { return true; }
                    }
                }
            }
            // Stmt::Let: we do not recurse into expression interiors here.
            Stmt::Let(_) => {}
            Stmt::Fn(f)
                if stmts_have_struct_match(&f.body, type_param_var_names, struct_field_names) => {
                    return true;
                }
            Stmt::If(i) => {
                for (_, branch) in &i.branches {
                    if stmts_have_struct_match(branch, type_param_var_names, struct_field_names) {
                        return true;
                    }
                }
                if let Some(eb) = &i.else_body {
                    if stmts_have_struct_match(eb, type_param_var_names, struct_field_names) {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}

/// Recursively scan an expression for `x is y` binary ops where both x and y are plain
/// variable names (not type names, not nil). Adds those variable names to `out`.
pub(crate) fn collect_is_identity_vars(
    expr: &Expr,
    type_names: &std::collections::HashSet<String>,
    out: &mut std::collections::HashSet<String>,
) {
    match &expr.kind {
        ExprKind::BinOp(BinOp::Is, l, r) | ExprKind::BinOp(BinOp::IsNot, l, r) => {
            // Only care about var is var (reference identity), not type/nil checks.
            if let (ExprKind::Var(lv), ExprKind::Var(rv)) = (&l.kind, &r.kind) {
                if !type_names.contains(lv.as_str()) && !type_names.contains(rv.as_str()) {
                    out.insert(lv.clone());
                    out.insert(rv.clone());
                }
            }
        }
        ExprKind::BinOp(_, l, r) => {
            collect_is_identity_vars(l, type_names, out);
            collect_is_identity_vars(r, type_names, out);
        }
        ExprKind::Call(callee, args) => {
            collect_is_identity_vars(callee, type_names, out);
            for a in args { collect_is_identity_vars(&a.value, type_names, out); }
        }
        ExprKind::UnaryOp(_, e) | ExprKind::Cast(e, _) => {
            collect_is_identity_vars(e, type_names, out);
        }
        ExprKind::If(if_stmt) => {
            for (cond, body) in &if_stmt.branches {
                collect_is_identity_vars(cond, type_names, out);
                for s in body { collect_is_identity_stmts(s, type_names, out); }
            }
            if let Some(eb) = &if_stmt.else_body {
                for s in eb { collect_is_identity_stmts(s, type_names, out); }
            }
        }
        _ => {}
    }
}

/// Infer a simple Rust type string for an expression argument inside Ok(...) / Err(...).
/// `param_tys` maps parameter names to their Rust type strings for variable lookup.
/// Returns None when the type cannot be determined from the expression alone.
pub(crate) fn infer_expr_type(expr: &Expr, param_tys: &std::collections::HashMap<String, String>) -> Option<String> {
    match &expr.kind {
        ExprKind::Int(_)   => Some("i64".to_string()),
        ExprKind::Float(_) => Some("f64".to_string()),
        ExprKind::Bool(_)  => Some("bool".to_string()),
        // String literals: `emit_expr_owned` wraps them in Arc<str>.
        ExprKind::Str(_) | ExprKind::StringInterp(_) => Some("Arc<str>".to_string()),
        ExprKind::Nil      => Some("()".to_string()),
        ExprKind::Void     => Some("()".to_string()),
        // Variable: look up param type
        ExprKind::Var(name) => param_tys.get(name).cloned(),
        // Binary numeric op: recurse on operands
        ExprKind::BinOp(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem, l, r) => {
            let lt = infer_expr_type(l, param_tys);
            let rt = infer_expr_type(r, param_tys);
            match (lt.as_deref(), rt.as_deref()) {
                (Some("i64"), _) | (_, Some("i64")) => Some("i64".to_string()),
                (Some("f64"), _) | (_, Some("f64")) => Some("f64".to_string()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Scans `stmts` for `return Ok(expr)` and `return Err(expr)` patterns.
/// Returns the inferred `(ok_type, err_type)` strings (or None when undetermined),
/// and whether each was found.
/// `param_tys` maps parameter names to their Rust type strings for variable lookup.
pub(crate) fn body_returns_result(stmts: &[Stmt], param_tys: &std::collections::HashMap<String, String>) -> (Option<String>, Option<String>) {
    let mut ok_ty: Option<String>  = None;
    let mut err_ty: Option<String> = None;
    for stmt in stmts {
        let (ok, err) = stmt_returns_result(stmt, param_tys);
        if ok_ty.is_none()  { ok_ty  = ok;  }
        if err_ty.is_none() { err_ty = err; }
    }
    (ok_ty, err_ty)
}

pub(crate) fn stmt_returns_result(stmt: &Stmt, param_tys: &std::collections::HashMap<String, String>) -> (Option<String>, Option<String>) {
    match stmt {
        Stmt::Return(ReturnStmt { value: Some(e), .. }) => {
            if let ExprKind::Call(callee, args) = &e.kind {
                if let ExprKind::Var(n) = &callee.kind {
                    let inner_ty = args.first()
                        .and_then(|a| infer_expr_type(&a.value, param_tys));
                    match n.as_str() {
                        "Ok"  => return (Some(inner_ty.unwrap_or_else(|| "()".to_string())), None),
                        "Err" => return (None, Some(inner_ty.unwrap_or_else(|| "Box<dyn std::error::Error + Send + Sync>".to_string()))),
                        _ => {}
                    }
                }
            }
            (None, None)
        }
        Stmt::If(i) => {
            let mut ok_ty  = None;
            let mut err_ty = None;
            for (_, body) in &i.branches {
                let (ok, err) = body_returns_result(body, param_tys);
                if ok_ty.is_none()  { ok_ty  = ok;  }
                if err_ty.is_none() { err_ty = err; }
            }
            if let Some(eb) = &i.else_body {
                let (ok, err) = body_returns_result(eb, param_tys);
                if ok_ty.is_none()  { ok_ty  = ok;  }
                if err_ty.is_none() { err_ty = err; }
            }
            (ok_ty, err_ty)
        }
        Stmt::While(w)    => body_returns_result(&w.body, param_tys),
        Stmt::WhileLet(w) => body_returns_result(&w.body, param_tys),
        Stmt::For(f)      => body_returns_result(&f.body, param_tys),
        Stmt::Fn(_)       => (None, None), // nested fn — don't scan inside
        _ => (None, None),
    }
}

/// Scan a statement for `is` reference identity comparisons.
/// Returns true if `Task.cancelled()` (no args) appears anywhere in the expression tree.
pub(crate) fn expr_uses_task_cancelled(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::MethodCall(obj, method, args) => {
            if method == "cancelled" && args.is_empty() {
                if let ExprKind::Var(v) = &obj.kind {
                    if v == "Task" { return true; }
                }
            }
            expr_uses_task_cancelled(obj) || args.iter().any(|a| expr_uses_task_cancelled(&a.value))
        }
        ExprKind::Call(callee, args) => {
            expr_uses_task_cancelled(callee) || args.iter().any(|a| expr_uses_task_cancelled(&a.value))
        }
        ExprKind::BinOp(_, l, r) => expr_uses_task_cancelled(l) || expr_uses_task_cancelled(r),
        ExprKind::UnaryOp(_, e) | ExprKind::Cast(e, _) => expr_uses_task_cancelled(e),
        ExprKind::Field(e, _) | ExprKind::OptionalField(e, _) => expr_uses_task_cancelled(e),
        ExprKind::Index(e, i) => expr_uses_task_cancelled(e) || expr_uses_task_cancelled(i),
        ExprKind::Array(elems) | ExprKind::Tuple(elems) | ExprKind::Set(elems) => {
            elems.iter().any(expr_uses_task_cancelled)
        }
        ExprKind::ArrayFill { value, count } => {
            expr_uses_task_cancelled(value) || expr_uses_task_cancelled(count)
        }
        ExprKind::ArrayAlloc { count } => expr_uses_task_cancelled(count),
        ExprKind::ArrayComp { expr, count, .. } => {
            expr_uses_task_cancelled(expr) || expr_uses_task_cancelled(count)
        }
        ExprKind::ArrayCompIter { expr, iter, .. } => {
            expr_uses_task_cancelled(expr) || expr_uses_task_cancelled(iter)
        }
        ExprKind::Block(stmts) | ExprKind::Do(stmts) => stmts_use_task_cancelled(stmts),
        ExprKind::Task(e) => expr_uses_task_cancelled(e),
        ExprKind::TaskWithTimeout(dur, body) => {
            expr_uses_task_cancelled(dur) || expr_uses_task_cancelled(body)
        }
        ExprKind::Else(e, d) | ExprKind::TryElse(e, d) => {
            expr_uses_task_cancelled(e) || expr_uses_task_cancelled(d)
        }
        ExprKind::TryElseBlock(try_stmts, else_stmts) => {
            stmts_use_task_cancelled(try_stmts) || stmts_use_task_cancelled(else_stmts)
        }
        _ => false,
    }
}

/// Returns true if `Task.cancelled()` appears anywhere in the statement list.
pub(crate) fn stmts_use_task_cancelled(stmts: &[Stmt]) -> bool {
    stmts.iter().any(stmt_uses_task_cancelled)
}

fn stmt_uses_task_cancelled(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Expr(e) => expr_uses_task_cancelled(e),
        Stmt::Let(l) => l.value.as_ref().is_some_and(expr_uses_task_cancelled),
        Stmt::Return(ReturnStmt { value: Some(e), .. })
        | Stmt::Throw(ThrowStmt { value: Some(e), .. }) => expr_uses_task_cancelled(e),
        Stmt::If(i) => {
            i.branches.iter().any(|(cond, body)| {
                expr_uses_task_cancelled(cond) || stmts_use_task_cancelled(body)
            }) || i.else_body.as_deref().is_some_and(stmts_use_task_cancelled)
        }
        Stmt::While(w) => {
            expr_uses_task_cancelled(&w.condition) || stmts_use_task_cancelled(&w.body)
        }
        Stmt::For(f) => {
            expr_uses_task_cancelled(&f.iterable) || stmts_use_task_cancelled(&f.body)
        }
        Stmt::Try(t) => {
            stmts_use_task_cancelled(&t.body)
                || t.catch_clauses.iter().any(|c| stmts_use_task_cancelled(&c.body))
        }
        Stmt::Defer(body) => stmts_use_task_cancelled(body),
        _ => false,
    }
}

pub(crate) fn collect_is_identity_stmts(
    stmt: &Stmt,
    type_names: &std::collections::HashSet<String>,
    out: &mut std::collections::HashSet<String>,
) {
    match stmt {
        Stmt::Expr(e) => {
            collect_is_identity_vars(e, type_names, out);
        }
        Stmt::Return(ReturnStmt { value: Some(e), .. }) => {
            collect_is_identity_vars(e, type_names, out);
        }
        Stmt::Let(l) => {
            if let Some(v) = &l.value { collect_is_identity_vars(v, type_names, out); }
        }
        Stmt::If(i) => {
            for (cond, body) in &i.branches {
                collect_is_identity_vars(cond, type_names, out);
                for s in body { collect_is_identity_stmts(s, type_names, out); }
            }
            if let Some(eb) = &i.else_body {
                for s in eb { collect_is_identity_stmts(s, type_names, out); }
            }
        }
        Stmt::Fn(f) => {
            for s in &f.body { collect_is_identity_stmts(s, type_names, out); }
        }
        _ => {}
    }
}

// ─── Overload mangling helpers ────────────────────────────────────────────────

/// Convert a Boring type to a short string for name mangling.
pub(crate) fn mangle_type_name(ty: &Type) -> String {
    match ty {
        Type::Int                  => "int".into(),
        Type::Uint                 => "uint".into(),
        Type::Float32                => "float32".into(),
        Type::Float64                => "float64".into(),
        Type::Bool                 => "bool".into(),
        Type::Str                  => "string".into(),
        Type::Void                 => "void".into(),
        Type::Array(inner)         => format!("arr_{}", mangle_type_name(inner)),
        Type::Optional(inner)      => format!("opt_{}", mangle_type_name(inner)),
        Type::Named(n)             => n.to_lowercase(),
        Type::Qualified(inner, _)  => mangle_type_name(inner),
        _                          => "t".into(),
    }
}

/// Build the mangled Rust function name for an overloaded function.
/// `describe(int n)` → `describe__int`
/// `process(int n, string s)` → `process__int__string`
pub(crate) fn mangle_overload_name(name: &str, params: &[crate::ast::Param]) -> String {
    let typed_params: Vec<&Type> = params.iter()
        .filter_map(|p| p.ty.as_ref())
        .collect();
    if typed_params.is_empty() {
        return name.to_string();
    }
    let suffix = typed_params.iter()
        .map(|t| mangle_type_name(t))
        .collect::<Vec<_>>()
        .join("__");
    format!("{}__{}", name, suffix)
}

/// Try to infer the Boring type of an expression for overload resolution.
/// Returns None when the type cannot be determined statically.
pub(crate) fn infer_overload_expr_type(
    expr: &Expr,
    var_types: &std::collections::HashMap<String, crate::ast::Type>,
    fn_return_types: &std::collections::HashMap<String, crate::ast::Type>,
    struct_fields: &std::collections::HashMap<String, Vec<(String, Type)>>,
) -> Option<Type> {
    match &expr.kind {
        ExprKind::Int(_)                              => Some(Type::Int),
        ExprKind::Float(_)                            => Some(Type::Float64),
        ExprKind::Bool(_)                             => Some(Type::Bool),
        ExprKind::Nil                                 => Some(Type::Optional(Box::new(Type::Void))),
        ExprKind::Str(_) | ExprKind::StringInterp(_) => Some(Type::Str),
        ExprKind::Array(_) | ExprKind::ArrayFill { .. } | ExprKind::ArrayAlloc { .. } | ExprKind::ArrayComp { .. } | ExprKind::ArrayCompIter { .. } => Some(Type::Array(Box::new(Type::Int))),
        ExprKind::Var(name) => var_types.get(name.as_str()).cloned(),
        ExprKind::Call(callee, _) => {
            if let ExprKind::Var(fn_name) = &callee.kind {
                fn_return_types.get(fn_name.as_str()).cloned()
            } else { None }
        }
        // Field access: look up the field type from struct_fields using the object's type.
        ExprKind::Field(obj_expr, field_name) => {
            let obj_ty = infer_overload_expr_type(obj_expr, var_types, fn_return_types, struct_fields)?;
            let struct_name = match &obj_ty {
                Type::Named(n) => Some(n.as_str()),
                Type::Qualified(inner, _) => if let Type::Named(n) = inner.as_ref() { Some(n.as_str()) } else { None },
                _ => None,
            }?;
            let fields = struct_fields.get(struct_name)?;
            fields.iter().find(|(fname, _)| fname == field_name).map(|(_, ft)| ft.clone())
        }
        _ => None,
    }
}

/// Check whether two boring types are compatible (for overload resolution).
pub(crate) fn types_compatible(expected: &Type, actual: &Type) -> bool {
    let expected = strip_qual_helper(expected);
    let actual = strip_qual_helper(actual);
    match (expected, actual) {
        (Type::Int,   Type::Int)   => true,
        (Type::Uint,  Type::Uint)  => true,
        (Type::Float32, Type::Float32) => true,
        (Type::Float64, Type::Float64) => true,
        (Type::Bool,  Type::Bool)  => true,
        (Type::Str,   Type::Str)   => true,
        (Type::Void,  Type::Void)  => true,
        (Type::Named(a), Type::Named(b)) => a == b,
        (Type::Named(n), t) | (t, Type::Named(n)) => match n.as_str() {
            "int"    => matches!(t, Type::Int),
            "uint"   => matches!(t, Type::Uint),
            "float32" | "f32" => matches!(t, Type::Float32),
            "float" | "float64" | "f64" => matches!(t, Type::Float64),
            "bool"   => matches!(t, Type::Bool),
            "string" => matches!(t, Type::Str),
            _ => false,
        },
        (Type::Array(a), Type::Array(b)) => types_compatible(a, b),
        (Type::Optional(a), Type::Optional(b)) => types_compatible(a, b),
        _ => false,
    }
}

/// Check whether two overload declarations conflict — i.e. there exists a call-arity N
/// such that both can be invoked with N arguments and all N parameter types match.
///
/// A function with default parameters can be called with fewer arguments than it declares,
/// which can create an ambiguous overlap with a shorter overload:
///
///   def fn(int n, string s = "x"):  # callable as fn(int) OR fn(int, string)
///   def fn(int n):                   # callable as fn(int)   ← CONFLICT at arity 1
///
/// Returns `Some(arity)` — the conflicting call-arity — or `None` if no conflict.
pub(crate) fn overloads_conflict(a: &crate::ast::FnDecl, b: &crate::ast::FnDecl) -> Option<usize> {
    // Minimum and maximum number of arguments each function accepts.
    let a_min = a.params.iter().filter(|p| p.default.is_none()).count();
    let b_min = b.params.iter().filter(|p| p.default.is_none()).count();
    let a_max = a.params.len();
    let b_max = b.params.len();

    // Iterate every arity that both functions can accept.
    let lo = a_min.max(b_min);
    let hi = a_max.min(b_max);
    for n in lo..=hi {
        // Check if types at every position are compatible.
        let conflict = a.params[..n].iter()
            .zip(b.params[..n].iter())
            .all(|(pa, pb)| match (&pa.ty, &pb.ty) {
                (Some(ta), Some(tb)) => types_compatible(ta, tb),
                _ => true, // untyped param matches anything
            });
        if conflict {
            return Some(n);
        }
    }
    None
}

fn strip_qual_helper(ty: &Type) -> &Type {
    match ty {
        Type::Qualified(inner, _) => strip_qual_helper(inner),
        other => other,
    }
}

// ── Reachability: which free functions does device code actually call? ────────
//
// Shared by cuda/rocm/metal's device emitters. Free functions are ordinary
// Boring functions shared with the host (CPU) build, and routinely use
// dynamic-array / heap constructs (`[float]` growable arrays, `.push`, string
// formatting, etc.) that have no device-C equivalent. Emitting every free
// function unconditionally -- regardless of whether any kernel calls it --
// makes the generated device file fail to compile as soon as the program has
// ANY host-only helper with this shape, even if no kernel ever touches it
// (confirmed for a plain CLI example with zero kernels at all). Restrict
// emission to the transitive closure of functions called from kernel entry
// points/methods instead.

/// Names of free (top-level, non-task, unqualified) functions transitively
/// called from any kernel's device methods/entry point. Only these should be
/// emitted into the device file.
pub(crate) fn reachable_free_fns(program: &Program) -> std::collections::HashSet<String> {
    use std::collections::HashMap;

    let free_fn_bodies: HashMap<&str, &[Stmt]> = program.items.iter().filter_map(|item| {
        if let Item::Fn(decl) = item {
            if decl.qualifier.is_none() && !decl.task {
                return Some((decl.name.as_str(), decl.body.as_slice()));
            }
        }
        None
    }).collect();

    let mut worklist: Vec<String> = Vec::new();
    for item in &program.items {
        if let Item::Kernel(decl) = item {
            for method in &decl.methods {
                collect_called_names(&method.body, &mut worklist);
            }
        }
    }

    let mut reachable = std::collections::HashSet::new();
    while let Some(name) = worklist.pop() {
        if !reachable.insert(name.clone()) { continue; }
        if let Some(body) = free_fn_bodies.get(name.as_str()) {
            collect_called_names(body, &mut worklist);
        }
    }
    reachable
}

fn collect_called_names(stmts: &[Stmt], out: &mut Vec<String>) {
    for stmt in stmts { collect_called_names_stmt(stmt, out); }
}

fn collect_called_names_stmt(stmt: &Stmt, out: &mut Vec<String>) {
    match stmt {
        Stmt::Let(s) => { if let Some(v) = &s.value { collect_called_names_expr(v, out); } }
        Stmt::Return(r) => { if let Some(v) = &r.value { collect_called_names_expr(v, out); } }
        Stmt::Expr(e) => collect_called_names_expr(e, out),
        Stmt::Throw(t) => { if let Some(v) = &t.value { collect_called_names_expr(v, out); } }
        Stmt::Break(_label, Some(v)) => collect_called_names_expr(v, out),
        Stmt::Wait(e, _) | Stmt::Yield(e, _) => collect_called_names_expr(e, out),
        Stmt::If(i) => {
            for (cond, body) in &i.branches {
                collect_called_names_expr(cond, out);
                collect_called_names(body, out);
            }
            if let Some(b) = &i.else_body { collect_called_names(b, out); }
        }
        Stmt::While(w) => { collect_called_names_expr(&w.condition, out); collect_called_names(&w.body, out); }
        Stmt::DoWhile(d) => { collect_called_names_expr(&d.condition, out); collect_called_names(&d.body, out); }
        Stmt::Loop(l) => collect_called_names(&l.body, out),
        Stmt::For(f) => { collect_called_names_expr(&f.iterable, out); collect_called_names(&f.body, out); }
        Stmt::Try(t) => {
            collect_called_names(&t.body, out);
            for c in &t.catch_clauses { collect_called_names(&c.body, out); }
        }
        Stmt::Defer(body) => collect_called_names(body, out),
        Stmt::Match(m) => {
            collect_called_names_expr(&m.subject, out);
            for arm in &m.arms {
                if let Some(g) = &arm.guard { collect_called_names_expr(g, out); }
                match &arm.body {
                    MatchBody::Expr(e) => collect_called_names_expr(e, out),
                    MatchBody::Block(body) => collect_called_names(body, out),
                }
            }
        }
        _ => {}
    }
}

fn collect_called_names_expr(expr: &Expr, out: &mut Vec<String>) {
    match &expr.kind {
        ExprKind::Call(callee, args) => {
            if let ExprKind::Var(name) = &callee.kind { out.push(name.clone()); }
            collect_called_names_expr(callee, out);
            for a in args { collect_called_names_expr(&a.value, out); }
        }
        ExprKind::GenericCall(callee, _, args) => {
            if let ExprKind::Var(name) = &callee.kind { out.push(name.clone()); }
            collect_called_names_expr(callee, out);
            for a in args { collect_called_names_expr(&a.value, out); }
        }
        ExprKind::Pipe(lhs, name, args) => {
            out.push(name.clone());
            collect_called_names_expr(lhs, out);
            for a in args { collect_called_names_expr(&a.value, out); }
        }
        ExprKind::MethodCall(obj, _, args) | ExprKind::OptionalMethodCall(obj, _, args) => {
            collect_called_names_expr(obj, out);
            for a in args { collect_called_names_expr(&a.value, out); }
        }
        ExprKind::BinOp(_, l, r) | ExprKind::Assign(l, r) | ExprKind::QuestionAssign(l, r)
        | ExprKind::Index(l, r) | ExprKind::Else(l, r) => {
            collect_called_names_expr(l, out);
            collect_called_names_expr(r, out);
        }
        ExprKind::UnaryOp(_, e) | ExprKind::Field(e, _) | ExprKind::Cast(e, _)
        | ExprKind::OptionalField(e, _) => collect_called_names_expr(e, out),
        ExprKind::If(i) => {
            for (cond, body) in &i.branches {
                collect_called_names_expr(cond, out);
                collect_called_names(body, out);
            }
            if let Some(b) = &i.else_body { collect_called_names(b, out); }
        }
        ExprKind::Match(m) => {
            collect_called_names_expr(&m.subject, out);
            for arm in &m.arms {
                if let Some(g) = &arm.guard { collect_called_names_expr(g, out); }
                match &arm.body {
                    MatchBody::Expr(e) => collect_called_names_expr(e, out),
                    MatchBody::Block(body) => collect_called_names(body, out),
                }
            }
        }
        ExprKind::Block(body) | ExprKind::Do(body) => collect_called_names(body, out),
        ExprKind::Array(items) | ExprKind::Tuple(items) | ExprKind::Set(items) => {
            for e in items { collect_called_names_expr(e, out); }
        }
        ExprKind::ArrayFill { value, count } => {
            collect_called_names_expr(value, out);
            collect_called_names_expr(count, out);
        }
        ExprKind::ArrayAlloc { count } => collect_called_names_expr(count, out),
        ExprKind::ArrayComp { expr, count, .. } => {
            collect_called_names_expr(expr, out);
            collect_called_names_expr(count, out);
        }
        ExprKind::ArrayCompIter { expr, iter, .. } => {
            collect_called_names_expr(expr, out);
            collect_called_names_expr(iter, out);
        }
        ExprKind::Dict(pairs) => {
            for (k, v) in pairs { collect_called_names_expr(k, out); collect_called_names_expr(v, out); }
        }
        ExprKind::Range { start, end, .. } => {
            collect_called_names_expr(start, out);
            collect_called_names_expr(end, out);
        }
        ExprKind::SliceRange { start, end, .. } => {
            if let Some(s) = start { collect_called_names_expr(s, out); }
            if let Some(e) = end { collect_called_names_expr(e, out); }
        }
        ExprKind::TryElse(a, b) => { collect_called_names_expr(a, out); collect_called_names_expr(b, out); }
        _ => {}
    }
}

// ── Consistent typed-error routing for enum `throw`s ────────────────────────
//
// `throws EnumName:` (typed) registers `EnumName` in `typed_error_enums`, which
// gates two independent things at emission time: (1) `emit_enum` auto-generates
// `Display`/`Error` impls for it, and (2) `emit_throw`'s `is_typed_error` check
// routes a `throw` of one of its variants through `BoringError::Other(TypeId,
// ...)` -- the only encoding a later `catch EnumName.Variant:` can downcast back
// out of. A bare `throws:` function throwing that SAME enum, when no `throws
// EnumName:` declaration exists ANYWHERE in the program, falls through instead
// to `BoringError::String(format!("{}", value))` -- stringifying the value via
// `Display` and permanently discarding which variant it was. Any `catch
// EnumName...:` clause elsewhere then silently never matches at runtime (the
// error surfaces as an "unhandled error" panic instead) -- confirmed via
// scratch-boring's `control_stop`/`StopSignal` (see docs/book.md's "Propagation
// across nested throws functions" section for the write-up).
//
// `typed_error_enums` is already global, per-program state -- consulted by
// `is_typed_error` regardless of which function is doing the current throwing
// (a single `throws EnumName:` declaration anywhere unlocks correct handling
// for every throw of that enum everywhere, typed function or not; confirmed by
// inspection of `emit_throw`). This closes the gap the other direction: any
// enum that is EVER the direct target of a `throw` statement -- typed
// `throws:` function or not -- is registered up front, so the correct,
// consistent `BoringError::Other` encoding (and the `Display`/`Error` impls
// that make it compile without a hand-written `as string:`) is used
// everywhere, matching the built-in `Error` enum's always-typed treatment.
// Purely additive: it only ever adds entries `is_typed_error` would otherwise
// have treated as "not typed", so no enum that already worked can regress.
pub(crate) fn collect_thrown_enum_names(
    program: &Program,
    all_enum_types: &std::collections::HashSet<String>,
    enum_variants: &std::collections::HashMap<String, String>,
    out: &mut std::collections::HashSet<String>,
) {
    fn visit_items(items: &[Item], all_enum_types: &std::collections::HashSet<String>, enum_variants: &std::collections::HashMap<String, String>, out: &mut std::collections::HashSet<String>) {
        for item in items {
            match item {
                Item::Fn(f) => collect_thrown_enum_names_stmts(&f.body, all_enum_types, enum_variants, out),
                Item::Struct(s) => {
                    for m in &s.methods { collect_thrown_enum_names_stmts(&m.body, all_enum_types, enum_variants, out); }
                    for i in &s.inits { collect_thrown_enum_names_stmts(&i.body, all_enum_types, enum_variants, out); }
                    for st in &s.setters { collect_thrown_enum_names_stmts(&st.body, all_enum_types, enum_variants, out); }
                    // Type-level methods (`type def`/`type req`/`type set`, called as
                    // `TypeName.method(...)`) were missing here -- an enum thrown ONLY from
                    // inside one of those bodies never got auto-registered as typed, unlike
                    // the same enum thrown from a regular instance method or free function.
                    for tm in &s.type_methods { collect_thrown_enum_names_stmts(&tm.body, all_enum_types, enum_variants, out); }
                }
                Item::Enum(e) => {
                    for m in &e.methods { collect_thrown_enum_names_stmts(&m.body, all_enum_types, enum_variants, out); }
                    for st in &e.setters { collect_thrown_enum_names_stmts(&st.body, all_enum_types, enum_variants, out); }
                    // Type-level methods (`type def`/`type req`/`type set`) -- mirrors the
                    // identical fix already applied to StructDecl::type_methods above.
                    for tm in &e.type_methods { collect_thrown_enum_names_stmts(&tm.body, all_enum_types, enum_variants, out); }
                }
                Item::Ext(e) => {
                    for m in &e.methods { collect_thrown_enum_names_stmts(&m.body, all_enum_types, enum_variants, out); }
                    for st in &e.setters { collect_thrown_enum_names_stmts(&st.body, all_enum_types, enum_variants, out); }
                }
                Item::Mod(m) => visit_items(&m.items, all_enum_types, enum_variants, out),
                Item::Stmt(s) => collect_thrown_enum_names_stmt(s, all_enum_types, enum_variants, out),
                _ => {}
            }
        }
    }
    visit_items(&program.items, all_enum_types, enum_variants, out);
}

fn collect_thrown_enum_names_stmts(stmts: &[Stmt], all_enum_types: &std::collections::HashSet<String>, enum_variants: &std::collections::HashMap<String, String>, out: &mut std::collections::HashSet<String>) {
    for stmt in stmts { collect_thrown_enum_names_stmt(stmt, all_enum_types, enum_variants, out); }
}

fn collect_thrown_enum_names_stmt(stmt: &Stmt, all_enum_types: &std::collections::HashSet<String>, enum_variants: &std::collections::HashMap<String, String>, out: &mut std::collections::HashSet<String>) {
    let rec = |body: &[Stmt], out: &mut std::collections::HashSet<String>| collect_thrown_enum_names_stmts(body, all_enum_types, enum_variants, out);
    match stmt {
        Stmt::Throw(t) => {
            if let Some(e) = &t.value {
                if let Some(name) = resolve_thrown_enum_name(e, all_enum_types, enum_variants) {
                    out.insert(name);
                }
            }
        }
        Stmt::If(i) => {
            for (_, body) in &i.branches { rec(body, out); }
            if let Some(b) = &i.else_body { rec(b, out); }
        }
        Stmt::While(w) => rec(&w.body, out),
        Stmt::DoWhile(d) => rec(&d.body, out),
        Stmt::Loop(l) => rec(&l.body, out),
        Stmt::For(f) => rec(&f.body, out),
        Stmt::Try(t) => {
            rec(&t.body, out);
            for c in &t.catch_clauses { rec(&c.body, out); }
        }
        Stmt::Defer(body) => rec(body, out),
        Stmt::Match(m) => {
            for arm in &m.arms {
                match &arm.body {
                    MatchBody::Expr(_) => {}
                    MatchBody::Block(body) => rec(body, out),
                }
            }
        }
        _ => {}
    }
}

/// Resolves a `throw`n expression to the user-defined enum type it targets, if
/// any -- mirroring the enum-specific patterns `emit_flow::emit_throw`'s
/// `is_typed_error` check recognizes (`EnumName.Variant`, bare `Variant`,
/// `EnumName.Variant(args)`, bare `Variant(args)`), but unconditionally (not
/// gated on the enum already being in `typed_error_enums` -- that set is
/// exactly what this function's result feeds into).
fn resolve_thrown_enum_name(e: &Expr, all_enum_types: &std::collections::HashSet<String>, enum_variants: &std::collections::HashMap<String, String>) -> Option<String> {
    match &e.kind {
        // `throw EnumName.Variant`
        ExprKind::Field(base, _variant) => match &base.kind {
            ExprKind::Var(n) if all_enum_types.contains(n.as_str()) => Some(n.clone()),
            _ => None,
        },
        // `throw Variant` (bare shorthand, resolved via the global variant→enum map)
        ExprKind::Var(n) => enum_variants.get(n.as_str()).cloned(),
        // `throw EnumName.Variant(args)` / `throw Variant(args)`
        ExprKind::Call(func, _) => match &func.kind {
            ExprKind::Field(base, _variant) => match &base.kind {
                ExprKind::Var(n) if all_enum_types.contains(n.as_str()) => Some(n.clone()),
                _ => None,
            },
            ExprKind::Var(n) => enum_variants.get(n.as_str()).cloned(),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod labeled_array_tests {
    use super::*;
    use crate::ast::{LabeledAxis, ConstExpr};

    fn int_axis(label: &str, n: i64) -> LabeledAxis {
        LabeledAxis {
            label: label.to_string(),
            size: Some(ConstExpr(Box::new(Expr { kind: ExprKind::Int(n), line: 0, col: 0, len: 0 }))),
        }
    }

    fn var_axis(label: &str, name: &str) -> LabeledAxis {
        LabeledAxis {
            label: label.to_string(),
            size: Some(ConstExpr(Box::new(Expr { kind: ExprKind::Var(name.to_string()), line: 0, col: 0, len: 0 }))),
        }
    }

    /// Row-major offset formula, 2 axes: `a0 + a1*d0`.
    #[test]
    fn labeled_array_at_index_row_major_offset_for_2d() {
        let axes = vec![int_axis("width", 800), int_axis("height", 600)];
        let labeled_args = vec![("width".to_string(), "10".to_string()), ("height".to_string(), "20".to_string())];
        let labeled_result = labeled_array_at_index(&axes, &labeled_args).expect("all labels present");

        assert_eq!(labeled_result, "10 + 20 * 800");
    }

    /// Row-major offset formula, 3 axes: `a0 + a1*d0 + a2*(d0*d1)`.
    #[test]
    fn labeled_array_at_index_row_major_offset_for_3d() {
        let axes = vec![int_axis("x", 2), int_axis("y", 3), int_axis("z", 4)];
        let labeled_args = vec![
            ("x".to_string(), "1".to_string()),
            ("y".to_string(), "2".to_string()),
            ("z".to_string(), "3".to_string()),
        ];
        let labeled_result = labeled_array_at_index(&axes, &labeled_args).expect("all labels present");

        assert_eq!(labeled_result, "1 + 2 * 2 + 3 * 6");
    }

    #[test]
    fn labeled_array_at_index_is_order_free_at_the_use_site() {
        let axes = vec![int_axis("width", 800), int_axis("height", 600)];
        let in_order = labeled_array_at_index(&axes, &[
            ("width".to_string(), "10".to_string()), ("height".to_string(), "20".to_string()),
        ]).unwrap();
        let reversed = labeled_array_at_index(&axes, &[
            ("height".to_string(), "20".to_string()), ("width".to_string(), "10".to_string()),
        ]).unwrap();
        assert_eq!(in_order, reversed);
    }

    #[test]
    fn labeled_array_at_index_none_when_a_label_is_missing() {
        let axes = vec![int_axis("width", 800), int_axis("height", 600)];
        let args = vec![("width".to_string(), "10".to_string())];
        assert_eq!(labeled_array_at_index(&axes, &args), None);
    }

    #[test]
    fn labeled_array_at_index_splices_in_a_const_generic_stride_unfolded() {
        // A LabeledArray axis may reference a kernel const-generic param —
        // the stride can't be folded to a single literal in that case.
        let axes = vec![int_axis("width", 800), var_axis("height", "H")];
        let args = vec![("width".to_string(), "10".to_string()), ("height".to_string(), "20".to_string())];
        let result = labeled_array_at_index(&axes, &args).unwrap();
        assert_eq!(result, "10 + 20 * 800");
    }

    #[test]
    fn labeled_array_dim_literal_stringifies_a_literal_axis() {
        let axes = vec![int_axis("width", 16), int_axis("height", 32)];
        assert_eq!(labeled_array_dim_literal(&axes, "width"), Some("16".to_string()));
        assert_eq!(labeled_array_dim_literal(&axes, "height"), Some("32".to_string()));
    }

    #[test]
    fn labeled_array_dim_literal_stringifies_a_const_generic_reference() {
        let axes = vec![var_axis("width", "W")];
        assert_eq!(labeled_array_dim_literal(&axes, "width"), Some("W".to_string()));
    }

    #[test]
    fn labeled_array_grid_dim_expr_ceil_divs_each_fixed_axis_for_2d() {
        let axes = vec![int_axis("width", 256), int_axis("height", 128)];
        assert_eq!(
            labeled_array_grid_dim_expr(&axes),
            "(((256 + block_dim.0 - 1) / block_dim.0), ((128 + block_dim.1 - 1) / block_dim.1), 1)",
        );
    }

    #[test]
    fn desugared_labeled_array_shadow_fields_finds_positional_siblings() {
        let fields = vec![
            KernelFieldDecl { name: "__src_axis0".into(), binding: FieldBinding::Let, qual: GpuQual::Const, ty: Type::Uint, default: None, line: 0, col: 0 },
            KernelFieldDecl { name: "__src_axis1".into(), binding: FieldBinding::Let, qual: GpuQual::Const, ty: Type::Uint, default: None, line: 0, col: 0 },
        ];
        assert_eq!(
            desugared_labeled_array_shadow_fields("src", &fields),
            Some(vec!["__src_axis0".to_string(), "__src_axis1".to_string()]),
        );
        assert_eq!(desugared_labeled_array_shadow_fields("other", &fields), None);
    }
}
