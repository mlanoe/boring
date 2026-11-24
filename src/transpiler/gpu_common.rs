// Free helpers that were byte-identical across the GPU backends (`cuda`, `rocm`, `metal`, `wgpu`).
// Each backend imports them with `use crate::transpiler::gpu_common::*;`; a backend that needs a
// different behaviour defines its own function of the same name, which shadows the glob import.

#![allow(unused_imports)]
use crate::ast::*;
use crate::transpiler::helpers::*;
use crate::transpiler::host_types::*;

pub(crate) fn const_array_count(e: &Expr) -> Option<usize> {
    match e.kind { ExprKind::Int(n) if n >= 0 => Some(n as usize), _ => None }
}

/// Argument names to pass at a call site for `field_params(fields)` — same filtering
/// and ordering, but just the bare field names instead of full C parameter declarations.
pub(crate) fn field_arg_names(fields: &[KernelFieldDecl]) -> Vec<String> {
    fields.iter().filter_map(|f| {
        match f.qual {
            GpuQual::Actor | GpuQual::Local => None,
            GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Surface => {
                Some(c_gpu_safe_ident(&f.name))
            }
            GpuQual::Const => {
                if matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _)) || f.ty.as_labeled_array().is_some() {
                    None  // __constant__ arrays accessed via file-scope global, not as args
                } else {
                    Some(c_gpu_safe_ident(&f.name))
                }
            }
        }
    }).chain(
        fields.iter().filter_map(|f| {
            match f.qual {
                GpuQual::Local => {
                    match &f.ty {
                        Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) => None,
                        ty if ty.as_labeled_array().is_some() => None,
                        _ => Some(c_gpu_safe_ident(&f.name)),
                    }
                }
                _ => None,
            }
        })
    ).collect()
}

pub(crate) fn binop_c(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",  BinOp::Sub => "-",  BinOp::Mul => "*",
        BinOp::Div => "/",  BinOp::Rem => "%",
        BinOp::Eq  => "==", BinOp::NotEq => "!=",
        BinOp::Lt  => "<",  BinOp::Gt => ">",
        BinOp::LtEq => "<=", BinOp::GtEq => ">=",
        BinOp::And => "&&", BinOp::Or => "||",
        BinOp::BitAnd => "&", BinOp::BitOr => "|", BinOp::BitXor => "^",
        BinOp::Shl => "<<", BinOp::Shr => ">>",
        _ => "/*op*/",
    }
}

pub(crate) fn unaryop_c(op: &UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg    => "-",
        UnaryOp::Not    => "!",
        UnaryOp::BitNot => "~",
    }
}

pub(crate) fn map_gpu_field(obj: &str, field: &str) -> String {
    match (obj, field) {
        ("gpu", "thread")    => "threadIdx".into(),
        ("gpu", "block")     => "blockIdx".into(),
        ("gpu", "block_dim" | "blockDim") => "blockDim".into(),
        ("gpu", "grid_dim" | "gridDim")  => "gridDim".into(),
        ("threadIdx", "x")   => "threadIdx.x".into(),
        ("threadIdx", "y")   => "threadIdx.y".into(),
        ("threadIdx", "z")   => "threadIdx.z".into(),
        ("blockIdx",  "x")   => "blockIdx.x".into(),
        ("blockIdx",  "y")   => "blockIdx.y".into(),
        ("blockIdx",  "z")   => "blockIdx.z".into(),
        ("blockDim",  "x")   => "blockDim.x".into(),
        ("blockDim",  "y")   => "blockDim.y".into(),
        ("blockDim",  "z")   => "blockDim.z".into(),
        ("gridDim",   "x")   => "gridDim.x".into(),
        ("gridDim",   "y")   => "gridDim.y".into(),
        ("gridDim",   "z")   => "gridDim.z".into(),
        ("gpu", "warp")      => "__warp".into(),
        ("__warp", "size")   => "warpSize".into(),
        ("__warp", "lane")   =>
            "((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) % warpSize)".into(),
        _                    => format!("{}.{}", obj, field),
    }
}

pub(crate) fn is_gpu_warp_receiver(obj: &Expr) -> bool {
    matches!(&obj.kind, ExprKind::Field(inner, name) if name == "warp"
        && matches!(&inner.kind, ExprKind::Var(v) if v == "gpu"))
}

pub(crate) fn body_has_explicit_sync(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Comment(c) if c == "sync" => true,
        Stmt::While(w)   => body_has_explicit_sync(&w.body),
        Stmt::For(f)     => body_has_explicit_sync(&f.body),
        Stmt::If(i)      => i.branches.iter().any(|(_, b)| body_has_explicit_sync(b))
                         || i.else_body.as_ref().is_some_and(|b| body_has_explicit_sync(b)),
        _ => false,
    })
}

pub(crate) fn body_accesses_sync_field(stmts: &[Stmt], fields: &[KernelFieldDecl]) -> bool {
    let sync_names: Vec<&str> = fields.iter()
        .filter(|f| matches!(f.qual, GpuQual::Actor))
        .map(|f| f.name.as_str())
        .collect();
    if sync_names.is_empty() { return false; }
    stmts_reference_any(stmts, &sync_names)
}

pub(crate) fn stmts_reference_any(stmts: &[Stmt], names: &[&str]) -> bool {
    stmts.iter().any(|s| stmt_references_any(s, names))
}

pub(crate) fn stmt_references_any(stmt: &Stmt, names: &[&str]) -> bool {
    match stmt {
        Stmt::Expr(e) => expr_references_any(e, names),
        Stmt::Return(r) => r.value.as_ref().is_some_and(|v| expr_references_any(v, names)),
        Stmt::Let(s) => s.value.as_ref().is_some_and(|v| expr_references_any(v, names)),
        Stmt::While(w) => stmts_reference_any(&w.body, names),
        Stmt::For(f)   => stmts_reference_any(&f.body, names),
        Stmt::If(i)    => i.branches.iter().any(|(_, b)| stmts_reference_any(b, names))
                       || i.else_body.as_ref().is_some_and(|b| stmts_reference_any(b, names)),
        _ => false,
    }
}

pub(crate) fn expr_references_any(expr: &Expr, names: &[&str]) -> bool {
    match &expr.kind {
        ExprKind::Var(n)           => names.contains(&n.as_str()),
        ExprKind::Index(a, i)      => expr_references_any(a, names) || expr_references_any(i, names),
        ExprKind::Field(e, _)      => expr_references_any(e, names),
        ExprKind::BinOp(_, l, r)   => expr_references_any(l, names) || expr_references_any(r, names),
        ExprKind::UnaryOp(_, e)    => expr_references_any(e, names),
        ExprKind::Assign(l, r)     => expr_references_any(l, names) || expr_references_any(r, names),
        ExprKind::Call(f, args)    => expr_references_any(f, names) || args.iter().any(|a| expr_references_any(&a.value, names)),
        _ => false,
    }
}

/// True when `ty` is one of boring's always-by-reference parameter kinds (see
/// `HostEmitter::fn_ref_params`'s doc comment for the exact contract this
/// mirrors).
pub(crate) fn is_ref_worthy_type(ty: &Type, struct_names: &std::collections::HashSet<String>) -> bool {
    match ty {
        Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) | Type::Dict(_, _) | Type::Set(_) => true,
        Type::Named(n) => struct_names.contains(n),
        // An explicit borrow (`File&`, `[uint8]&`) always renders as a Rust `&`/`&mut`
        // parameter, regardless of what it wraps -- matches `metal::host`'s identical
        // arm on its own (not yet shared) copy of this function.
        Type::Qualified(_, OwnerQual::Borrow | OwnerQual::BorrowMut) => true,
        _ => false,
    }
}

/// True for a Boring `string` type, under either spelling (`Type::Str` from a
/// `string`-typed `let`/field, or `Type::Named("string"|"str")` from a parameter/
/// return-type annotation). Matches `metal::host`'s identical (not yet shared)
/// free function of the same name.
pub(crate) fn is_string_type(ty: &Type) -> bool {
    matches!(ty.without_mut(), Type::Str)
        || matches!(ty.without_mut(), Type::Named(n) if n == "string" || n == "str")
}

/// Per-parameter by-ref/by-mut flags, plus GPU-resident-return info, scanned once
/// from a single `Item::Fn` — shared by the ROCm and CUDA restricted host emitters'
/// whole-program `Item::Fn` pre-pass (`fn_ref_params`/`fn_mut_params`/
/// `fn_returns_resident`/`fn_returns_resident_tuple`, each backend's own doc comment
/// explains why this must be whole-program and agree across emitters). `resident`
/// and `resident_tuple` are mutually exclusive: a `Type::Tuple` never itself
/// satisfies `gpu_resident_qual`, only its elements can.
pub(crate) struct FnCallConvention {
    pub ref_flags: Vec<bool>,
    pub mut_flags: Vec<bool>,
    pub resident: Option<Type>,
    pub resident_tuple: Option<Vec<bool>>,
}

pub(crate) fn scan_fn_call_convention(f: &FnDecl, struct_names: &std::collections::HashSet<String>) -> FnCallConvention {
    let ref_flags = f.params.iter()
        .map(|p| p.ty.as_ref().is_some_and(|ty| is_ref_worthy_type(ty, struct_names)))
        .collect();
    let mut_flags = f.params.iter().map(Param::grants_content_mut).collect();
    let (resident, resident_tuple) = match &f.return_ty {
        Some(rt) if rt.gpu_resident_qual().is_some() => (Some(rt.clone()), None),
        Some(Type::Tuple(elems)) => {
            let flags: Vec<bool> = elems.iter().map(|t| t.gpu_resident_qual().is_some()).collect();
            if flags.iter().any(|&resident| resident) { (None, Some(flags)) } else { (None, None) }
        }
        _ => (None, None),
    };
    FnCallConvention { ref_flags, mut_flags, resident, resident_tuple }
}

/// Coerce an already-rendered argument expression (`rendered`, from the emitter's own
/// `self.expr(arg)`) to match the callee's declared by-ref/owned parameter convention
/// (`fn_ref_params`/`fn_mut_params`). `is_ref_var` is true when the argument
/// expression is itself a tracked already-by-reference local (`self.ref_params`), so
/// passing it through unchanged is correct either way. Shared by the ROCm and CUDA
/// restricted host emitters' `coerce_call_arg_mut`.
pub(crate) fn coerce_call_arg(rendered: String, is_ref_var: bool, callee_expects_ref: bool, wants_mut: bool) -> String {
    match (callee_expects_ref, is_ref_var) {
        (true, true) => rendered,
        (true, false) => if wants_mut { format!("&mut ({rendered})") } else { format!("&({rendered})") },
        (false, true) => format!("({rendered}).clone()"),
        (false, false) => rendered,
    }
}

/// The Rust return type for a function whose return type may itself be GPU-resident
/// (`resident_elem`, already rendered to its element type string) or a `Tuple` with
/// some resident elements (`resident_tuple`, from `FnCallConvention`) — wrapping
/// either the whole type or just the resident tuple positions in
/// `BoringKernelArg<T>`. `elem_ty`/`base_ty` are the backend's own
/// `elem_rust_type`/`rust_type` free functions (passed through rather than called
/// directly here, since a backend needing a different host-element convention —
/// e.g. Metal's `general_host_elem_type` bridge — can still share this shape).
/// Shared by the ROCm and CUDA restricted host emitters' `emit_fn`.
pub(crate) fn resident_return_type(
    return_ty: Option<&Type>,
    resident_elem: Option<&str>,
    resident_tuple: Option<&Vec<bool>>,
    elem_ty: fn(&Type) -> String,
    base_ty: fn(&Type) -> String,
) -> String {
    if let Some(elem) = resident_elem {
        return format!("BoringKernelArg<{elem}>");
    }
    match (return_ty, resident_tuple) {
        (Some(Type::Tuple(elems)), Some(flags)) => {
            let parts: Vec<String> = elems.iter().enumerate().map(|(i, ty)| {
                if flags.get(i).copied().unwrap_or(false) {
                    format!("BoringKernelArg<{}>", elem_ty(ty))
                } else {
                    base_ty(ty)
                }
            }).collect();
            format!("({})", parts.join(", "))
        }
        _ => return_ty.map(base_ty).unwrap_or_else(|| "()".into()),
    }
}

/// `print`/`error`/`warn`/`info`/`debug`/`trace` called from a GPU host function: the
/// Rust macro the standard transpiler (`emit_expr.rs`) emits for the same builtin, given
/// the already-rendered first argument (`None` for a zero-argument call). Log levels
/// print `[LEVEL] message` on stderr, like `boring run`. `None` for any other name.
pub(crate) fn host_display_builtin(name: &str, inner: Option<String>) -> Option<String> {
    let (mac, prefix) = match name {
        "print" => ("println", String::new()),
        "error" | "warn" | "info" | "debug" | "trace" => ("eprintln", format!("[{}] ", name.to_uppercase())),
        _ => return None,
    };
    let Some(inner) = inner else {
        return Some(if prefix.is_empty() { format!("{mac}!()") } else { format!("{mac}!(\"{}\")", prefix.trim_end()) });
    };
    // A rendered interpolation (`format!("...", args)`) lifts into the macro itself.
    if let Some(inner_args) = inner.strip_prefix("format!(\"").and_then(|s| s.strip_suffix(')')) {
        return Some(format!("{mac}!(\"{prefix}{inner_args})"));
    }
    Some(format!("{mac}!(\"{prefix}{{}}\", {inner})"))
}

/// Splits a Rust parameter list (`a: &Vec<f32>, b: (isize, f32)`) at its top-level commas.
fn split_rust_params(params: &str) -> Vec<String> {
    let (mut out, mut cur, mut depth) = (Vec::new(), String::new(), 0i32);
    for c in params.chars() {
        match c {
            '(' | '<' | '[' => { depth += 1; cur.push(c) }
            ')' | '>' | ']' => { depth -= 1; cur.push(c) }
            ',' if depth == 0 => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() { out.push(cur); }
    out
}

/// The Rust parameter types of free function `name` as the general pass emitted it in `code`
/// (empty when `code` has no such function, e.g. a kernel-touching one the host emits itself).
pub(crate) fn general_fn_param_types(code: &str, name: &str) -> Vec<String> {
    let needle = format!("fn {name}(");
    let Some(sig) = code.lines().find_map(|l| {
        let t = l.trim_start();
        let t = t.strip_prefix("pub ").unwrap_or(t);
        t.starts_with(&needle).then(|| t.to_string())
    }) else { return Vec::new() };
    let rest = &sig[needle.len()..];
    let mut depth = 0i32;
    let end = rest.char_indices().find_map(|(i, c)| match c {
        '(' | '<' | '[' => { depth += 1; None }
        ')' if depth == 0 => Some(i),
        ')' | '>' | ']' => { depth -= 1; None }
        _ => None,
    }).unwrap_or(rest.len());
    split_rust_params(&rest[..end]).iter()
        .map(|p| p.split_once(':').map(|(_, t)| t.trim().to_string()).unwrap_or_default())
        .collect()
}

/// Per parameter of free function `name`, `Some(takes_ref)` when the general pass emitted it as
/// plain host data (no `BoringKernelArg`). Residency inference may tag a host-only array
/// parameter `'kernel'global`; the general pass drops that qualifier again
/// (`unqualify_host_only_gpu_array_params`), so a host caller must not wrap the argument in
/// `BoringKernelArg::Host(..)` for it. `None` for a position (or function) it didn't emit.
pub(crate) fn general_plain_params(code: &str, name: &str) -> Vec<Option<bool>> {
    general_fn_param_types(code, name).iter()
        .map(|t| (!t.is_empty() && !t.contains("BoringKernelArg")).then(|| t.starts_with('&')))
        .collect()
}

/// How the general pass emitted method `method` of struct `ty` in `code` (its `impl ty` block):
/// one passing mode per non-`self` parameter (`"&mut"`, `"&"`, `"box"` for `Box<dyn ..>`,
/// `"f64"`/`"f32"` for a float scalar, `""` by value) and whether it returns a `Result` (a
/// `throws` method, which a host caller `?`-propagates). `None` when no such method is found.
pub(crate) fn general_impl_method_sig(code: &str, ty: &str, method: &str) -> Option<(Vec<&'static str>, bool)> {
    let mut in_impl = false;
    for l in code.lines() {
        if l.trim().is_empty() { continue; }
        if !l.starts_with(char::is_whitespace) {
            let t = l.trim_end();
            in_impl = t.starts_with("impl ") && t.trim_end_matches(" {").split_whitespace().last() == Some(ty)
                || t.starts_with("impl<") && t.trim_end_matches(" {").ends_with(&format!(" {ty}"));
            continue;
        }
        if !in_impl { continue; }
        let t = l.trim_start();
        let t = t.strip_prefix("pub ").unwrap_or(t);
        let Some(rest) = t.strip_prefix(&format!("fn {method}(")) else { continue };
        let mut depth = 0i32;
        let mut end = None;
        for (i, c) in rest.char_indices() {
            match c { '(' | '<' | '[' => depth += 1, ')' | '>' | ']' if depth > 0 => depth -= 1, ')' => { end = Some(i); break; } _ => {} }
        }
        let end = end?;
        let mut modes: Vec<&'static str> = split_rust_params(&rest[..end]).iter().map(|p| {
            let ty = p.split_once(':').map(|(_, ty)| ty.trim()).unwrap_or("");
            if ty.starts_with("&mut ") { "&mut" } else if ty.starts_with('&') { "&" }
            else if ty.starts_with("Box<dyn ") { "box" } else if ty == "f64" { "f64" } else if ty == "f32" { "f32" } else { "" }
        }).collect();
        if modes.is_empty() { return None; }
        modes.remove(0); // `self`
        let returns_result = rest[end + 1..].trim_start().strip_prefix("->")
            .is_some_and(|r| r.trim_start().starts_with("Result<"));
        return Some((modes, returns_result));
    }
    None
}

pub(crate) fn is_gpu_array_param(ty: &Type) -> bool {
    matches!(ty.without_mut(), Type::Qualified(inner, OwnerQual::GpuGlobal | OwnerQual::GpuUnified)
        if matches!(inner.without_mut(), Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) | Type::LabeledArray(_, _)))
}

/// Returns the size in bytes of a scalar element type for shared-memory calculations.
pub(crate) fn literal_array_uses_float16(expr: &Expr) -> bool {
    fn is_half(expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Call(callee, args) if args.len() == 1 =>
                matches!(&callee.kind, ExprKind::Var(name) if matches!(name.as_str(), "float16" | "f16")),
            ExprKind::Cast(_, ty) => matches!(ty.without_mut(), Type::Float16)
                || matches!(ty.without_mut(), Type::Named(name) if matches!(name.as_str(), "float16" | "f16")),
            _ => false,
        }
    }
    matches!(&expr.kind, ExprKind::Array(elements) if elements.iter().any(is_half))
}

pub(crate) fn binop_rust(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",   BinOp::Sub => "-",  BinOp::Mul => "*",
        BinOp::Div => "/",   BinOp::Rem => "%",
        BinOp::Eq  => "==",  BinOp::NotEq => "!=",  BinOp::RefEq => "==",
        BinOp::Lt  => "<",   BinOp::Gt => ">",
        BinOp::LtEq => "<=", BinOp::GtEq => ">=",
        BinOp::And => "&&",  BinOp::Or => "||",
        BinOp::BitAnd => "&", BinOp::BitOr => "|", BinOp::BitXor => "^",
        BinOp::Shl => "<<",  BinOp::Shr => ">>",
        _ => "/*op*/",
    }
}

pub(crate) fn unaryop_rust(op: &UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg    => "-",
        UnaryOp::Not    => "!",
        UnaryOp::BitNot => "!",
    }
}
