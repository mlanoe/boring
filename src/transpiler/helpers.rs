use super::*;

pub(crate) fn looks_like_collection(expr: &str) -> bool {
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

/// Normalize boring primitive type names (lowercase aliases) to Rust equivalents.
/// Pass `use_rc = true` in single-thread mode so `string` maps to `Rc<str>` instead of `Arc<str>`.
pub(crate) fn normalize_type_name(name: &str, use_rc: bool) -> String {
    match name {
        "string"            => if use_rc { "Rc<str>".into() } else { "Arc<str>".into() },
        "str"               => "&str".into(),
        "String"            => "String".into(),
        "int"    | "Int"    => "i64".into(),
        "uint"   | "Uint"   => "u64".into(),
        "float"  | "Float"  => "f64".into(),
        "bool"   | "Bool"   => "bool".into(),
        "void"   | "Void"   => "()".into(),
        "nil"    | "Nil"    => "()".into(),
        "never"  | "Never"  => "!".into(),
        // Rust numeric aliases pass through unchanged
        "i8" | "i16" | "i32" | "i64" | "isize" => name.into(),
        "u8" | "u16" | "u32" | "u64" | "usize" => name.into(),
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

/// Map boring method names to (rust_method, optional_suffix).
pub(crate) fn map_method(name: &str, _arity: usize) -> (String, Option<&'static str>) {
    match name {
        // len() returns usize; Boring's length/count returns int (i64).
        "length" | "count" => ("len".into(), Some(" as i64")),
        // len() called directly (not via length/count) — cast to u64 so comparisons
        // with Boring's `uint` (u64) variables don't cause type mismatch errors.
        "len"              => ("len".into(), Some(" as u64")),
        "isEmpty"          => ("is_empty".into(), None),
        "push"             => ("push".into(), None),
        // Vec::pop() returns Option<T>; unwrap to match Boring semantics (returns the value or default).
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
        "parse_int"        => ("parse::<i64>().ok".into(), Some("")),
        "parse_float"      => ("parse::<f64>().ok".into(), Some("")),
        "toUpperCase" | "uppercased" | "upper" | "to_upper" | "toUpper" => ("to_uppercase".into(), None),
        "toLowerCase" | "lowercased" | "lower" | "to_lower" | "toLower" => ("to_lowercase".into(), None),
        "startsWith" | "hasPrefix"   => ("starts_with".into(), None),
        "endsWith"   | "hasSuffix"   => ("ends_with".into(), None),
        "first"            => ("first".into(), None),
        "last"             => ("last".into(), None),
        "append"           => ("push".into(), None),
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

/// Returns true when an expression is likely to produce an `Option<T>` value,
/// used to avoid wrapping Option-chain methods in `.iter().cloned().collect()`.
pub(crate) fn is_option_expr(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::MethodCall(recv, method, _) | ExprKind::Pipe(recv, method, _) => {
            // Unambiguously Option-producing (not Vec methods):
            const OPTION_ONLY: &[&str] = &["and_then", "or_else", "flatten", "as_ref", "as_deref"];
            if OPTION_ONLY.contains(&method.as_str()) { return true; }
            // Ambiguous (exists on both Vec and Option): propagate — only treat as
            // Option if the receiver is itself Option-like.
            const MAYBE_OPTION: &[&str] = &[
                "filter", "map", "next", "cloned", "copied",
                "find", "first", "last", "get", "pop",
            ];
            if MAYBE_OPTION.contains(&method.as_str()) {
                return is_option_expr(recv);
            }
            false
        }
        ExprKind::Var(name) => name.ends_with('?'),
        _ => false,
    }
}

/// Map boring field names to Rust field names.
pub(crate) fn map_field(name: &str) -> &str {
    match name {
        // len() returns usize in Rust; Boring's `int` is i64 — cast so the type matches.
        "length" | "count" => "len() as i64",
        "isEmpty" => "is_empty()",
        other => other,
    }
}

/// Map a Boring type name to a BoringError match arm pattern and the `error: Arc<str>` binding.
/// Returns (arm_pattern, error_binding_expr).
/// Returns (arm_pattern, error_arc_binding) for each BoringError variant that matches `ty`.
/// String types produce two entries (Str for literals, String for dynamic).
pub(crate) fn boring_type_to_boring_val_arms(ty: &str) -> Vec<(String, String)> {
    match ty {
        "String" | "string" | "cstring" | "tstring" => vec![
            // &'static str literal
            ("BoringError::Str(__boring_s)".to_string(),
             "Arc::<str>::from(__boring_s.to_string())".to_string()),
            // Arc<str> from interpolation or re-throw
            ("BoringError::String(ref __boring_s)".to_string(),
             "__boring_s.clone()".to_string()),
        ],
        "Int" | "int" => vec![
            ("BoringError::Int(__boring_n)".to_string(),
             "Arc::<str>::from(__boring_n.to_string())".to_string()),
        ],
        "Float" | "float" => vec![
            ("BoringError::Float(__boring_f)".to_string(),
             "Arc::<str>::from(__boring_f.to_string())".to_string()),
        ],
        "Bool" | "bool" => vec![
            ("BoringError::Bool(__boring_b)".to_string(),
             "Arc::<str>::from(__boring_b.to_string())".to_string()),
        ],
        other => vec![
            // Unknown type: will be handled by the named-clause path (BoringError::Other)
            (format!("/* unreachable catch {} */ ref __boring_other", other),
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

pub(crate) fn collect_vars_in(expr: &Expr, out: &mut Vec<String>) {
    match &expr.kind {
        ExprKind::Var(name)                 => out.push(name.clone()),
        ExprKind::BinOp(_, l, r)            => { collect_vars_in(l, out); collect_vars_in(r, out); }
        ExprKind::UnaryOp(_, e)             => collect_vars_in(e, out),
        ExprKind::Field(e, _) | ExprKind::OptionalField(e, _) => collect_vars_in(e, out),
        ExprKind::Index(e, i)               => { collect_vars_in(e, out); collect_vars_in(i, out); }
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

        // match expression — walk subject and each arm (guard + body)
        ExprKind::Match(match_stmt) => {
            collect_vars_in(&match_stmt.subject, out);
            for arm in &match_stmt.arms {
                if let Some(guard) = &arm.guard { collect_vars_in(guard, out); }
                match &arm.body {
                    MatchBody::Expr(e) => collect_vars_in(e, out),
                    MatchBody::Block(stmts) => {
                        for s in stmts { collect_vars_in_stmt(s, out); }
                    }
                }
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
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Str(_) | ExprKind::Bool(_)
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
                if let Some(guard) = &arm.guard { collect_vars_in(guard, out); }
                match &arm.body {
                    MatchBody::Expr(e) => collect_vars_in(e, out),
                    MatchBody::Block(stmts) => {
                        for s in stmts { collect_vars_in_stmt(s, out); }
                    }
                }
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
            Stmt::If(i) => {
                if i.branches.iter().any(|(_, body)| body_has_early_return(body))
                    || i.else_body.as_deref().map_or(false, body_has_early_return)
                {
                    return true;
                }
            }
            Stmt::While(w) => { if body_has_early_return(&w.body) { return true; } }
            Stmt::For(f)   => { if body_has_early_return(&f.body) { return true; } }
            Stmt::WhileLet(w) => { if body_has_early_return(&w.body) { return true; } }
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

/// Returns `true` if any top-level item in the list contains a `task expr` (either as a
/// standalone detached task statement or as the RHS of a `let` binding).  Used by
/// `emit_program` to decide whether the auto-generated `main` needs to be `async`.
/// Returns true if any statement in the body declares a T'actor or T'guard binding, meaning
/// the function will implicitly generate `.lock().await` / `.read().await` calls and needs to be async.
#[allow(dead_code)]
pub(crate) fn body_has_actor_binding(stmts: &[Stmt]) -> bool {
    for stmt in stmts {
        match stmt {
            Stmt::Let(l) => {
                if let Some(ty) = &l.ty {
                    if Transpiler::is_mutex_binding(l.binding.is_mutable(), ty) {
                        return true;
                    }
                    if Transpiler::is_rwlock_binding(l.binding.is_mutable(), ty) {
                        return true;
                    }
                }
                // Recurse into nested blocks
                if l.value.as_ref().map_or(false, |v| expr_has_actor_binding(v)) { return true; }
            }
            Stmt::If(i) => {
                if i.branches.iter().any(|(_, body)| body_has_actor_binding(body))
                    || i.else_body.as_deref().map_or(false, body_has_actor_binding)
                {
                    return true;
                }
            }
            Stmt::While(w) => { if body_has_actor_binding(&w.body) { return true; } }
            Stmt::For(f)   => { if body_has_actor_binding(&f.body) { return true; } }
            Stmt::Defer(b) => { if body_has_actor_binding(b) { return true; } }
            _ => {}
        }
    }
    false
}

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
            Stmt::If(i) => {
                if i.branches.iter().any(|(_, body)| body_has_stream_for(body, stream_fns))
                    || i.else_body.as_deref().map_or(false, |b| body_has_stream_for(b, stream_fns))
                { return true; }
            }
            Stmt::While(w)   => { if body_has_stream_for(&w.body, stream_fns)  { return true; } }
            Stmt::Defer(b)   => { if body_has_stream_for(b,        stream_fns)  { return true; } }
            _ => {}
        }
    }
    false
}

#[allow(dead_code)]
pub(crate) fn expr_has_actor_binding(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Block(stmts) => body_has_actor_binding(stmts),
        _ => false,
    }
}

/// Returns true if any `channel<T>(n)` call or `task:` expression appears in `stmts`.
/// Used to detect that `main` (or another function) needs to be async.
pub(crate) fn body_has_channel_or_task(stmts: &[Stmt]) -> bool {
    for stmt in stmts {
        match stmt {
            Stmt::Let(l) => {
                if l.value.as_ref().map_or(false, |v| expr_has_channel_or_task(v)) { return true; }
            }
            Stmt::Expr(e) | Stmt::Return(ReturnStmt { value: Some(e), .. }) => {
                if expr_has_channel_or_task(e) { return true; }
            }
            Stmt::If(i) => {
                if i.branches.iter().any(|(_, b)| body_has_channel_or_task(b))
                    || i.else_body.as_deref().map_or(false, body_has_channel_or_task)
                { return true; }
            }
            Stmt::While(w) => { if body_has_channel_or_task(&w.body) { return true; } }
            Stmt::For(f)   => { if body_has_channel_or_task(&f.body) { return true; } }
            Stmt::Defer(b) => { if body_has_channel_or_task(b) { return true; } }
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
            Stmt::Expr(e) | Stmt::Return(ReturnStmt { value: Some(e), .. }) => {
                if expr_calls_task_fn(e, task_fns) { return true; }
            }
            Stmt::Let(l) => {
                if l.value.as_ref().map_or(false, |v| expr_calls_task_fn(v, task_fns)) { return true; }
            }
            Stmt::If(i) => {
                if i.branches.iter().any(|(_, b)| body_calls_task_fn(b, task_fns))
                    || i.else_body.as_deref().map_or(false, |b| body_calls_task_fn(b, task_fns))
                { return true; }
            }
            Stmt::While(w) => { if body_calls_task_fn(&w.body, task_fns) { return true; } }
            Stmt::For(f)   => { if body_calls_task_fn(&f.body, task_fns) { return true; } }
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
            Stmt::If(i) => {
                if i.branches.iter().any(|(_, b)| body_has_wait(b))
                    || i.else_body.as_deref().map_or(false, body_has_wait)
                { return true; }
            }
            Stmt::While(w) => { if body_has_wait(&w.body) { return true; } }
            Stmt::For(f)   => { if body_has_wait(&f.body) { return true; } }
            Stmt::Defer(b) => { if body_has_wait(b)       { return true; } }
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
            Item::Stmt(s) => { if stmt_has_task(s) { return true; } }
            Item::Let(l)  => { if l.value.as_ref().map_or(false, |v| expr_has_task(v)) { return true; } }
            _ => {}
        }
    }
    false
}

pub(crate) fn items_have_task_call(items: &[&Item], task_fns: &std::collections::HashSet<String>) -> bool {
    for item in items {
        match item {
            Item::Stmt(s) => {
                if let Stmt::Expr(e) = s {
                    if expr_has_task_call(e, task_fns) { return true; }
                }
            }
            Item::Let(l) => { if l.value.as_ref().map_or(false, |v| expr_has_task_call(v, task_fns)) { return true; } }
            _ => {}
        }
    }
    false
}

pub(crate) fn stmt_has_task(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Expr(e) => expr_has_task(e),
        Stmt::Let(l)  => l.value.as_ref().map_or(false, |v| expr_has_task(v)),
        Stmt::If(i) => {
            i.branches.iter().any(|(cond, body)| {
                expr_has_task(cond) || body.iter().any(stmt_has_task)
            }) || i.else_body.as_deref().map_or(false, |b| b.iter().any(stmt_has_task))
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

/// Returns true if the Rust type string is a specific numeric type that may need coercion.
pub(crate) fn is_specific_numeric_type(ty: &str) -> bool {
    matches!(ty, "i8" | "i16" | "i32" | "i64" | "isize"
               | "u8" | "u16" | "u32" | "u64" | "usize"
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
            "usize" => (0, 64),
            "i8"    => (1, 8),
            "i16"   => (1, 16),
            "i32"   => (1, 32),
            "i64"   => (1, 64),
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
            Stmt::Fn(f) => {
                if stmts_have_struct_match(&f.body, type_param_var_names, struct_field_names) {
                    return true;
                }
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
        Stmt::Let(l) => l.value.as_ref().map_or(false, |v| expr_uses_task_cancelled(v)),
        Stmt::Return(ReturnStmt { value: Some(e), .. })
        | Stmt::Throw(ThrowStmt { value: Some(e), .. }) => expr_uses_task_cancelled(e),
        Stmt::If(i) => {
            i.branches.iter().any(|(cond, body)| {
                expr_uses_task_cancelled(cond) || stmts_use_task_cancelled(body)
            }) || i.else_body.as_deref().map_or(false, stmts_use_task_cancelled)
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
        Type::Float                => "float".into(),
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
        ExprKind::Float(_)                            => Some(Type::Float),
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
        (Type::Float, Type::Float) => true,
        (Type::Bool,  Type::Bool)  => true,
        (Type::Str,   Type::Str)   => true,
        (Type::Void,  Type::Void)  => true,
        (Type::Named(a), Type::Named(b)) => a == b,
        (Type::Named(n), t) | (t, Type::Named(n)) => match n.as_str() {
            "int"    => matches!(t, Type::Int),
            "uint"   => matches!(t, Type::Uint),
            "float"  => matches!(t, Type::Float),
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
