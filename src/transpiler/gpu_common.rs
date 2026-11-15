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
        _ => false,
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
