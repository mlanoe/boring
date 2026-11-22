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
