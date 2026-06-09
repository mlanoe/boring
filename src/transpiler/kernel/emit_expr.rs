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

// Expression emission for the kernel transpiler.

use crate::ast::{Expr, ExprKind, BinOp, UnaryOp};
use super::helpers::KernelTranspiler;

fn escape_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

fn binop_str(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add  => "+",
        BinOp::Sub  => "-",
        BinOp::Mul  => "*",
        BinOp::Div  => "/",
        BinOp::Rem  => "%",
        BinOp::Eq   => "==",
        BinOp::NotEq => "!=",
        BinOp::RefEq => "==",
        BinOp::Lt   => "<",
        BinOp::Gt   => ">",
        BinOp::LtEq => "<=",
        BinOp::GtEq => ">=",
        BinOp::And  => "&&",
        BinOp::Or   => "||",
        BinOp::BitAnd => "&",
        BinOp::BitOr  => "|",
        BinOp::BitXor => "^",
        BinOp::Shl  => "<<",
        BinOp::Shr  => ">>",
        BinOp::Is | BinOp::IsNot => "==",
    }
}

impl KernelTranspiler {
    pub(super) fn emit_expr(&self, expr: &Expr) -> String {
        match &expr.kind {
            ExprKind::Int(n)   => n.to_string(),
            ExprKind::Float(f) => {
                // floats are not supported in kernel code; emit a comment
                format!("/* float forbidden */ {}", f)
            }
            // String literals: use c_str! macro for &CStr (preferred in kernel)
            ExprKind::Str(s)   => {
                // c_str! produces &'static CStr; for owned CString use try_from_fmt
                format!("c_str!(\"{}\")", escape_str(s))
            }
            ExprKind::StringInterp(segs) => {
                // Build a format string; kernel uses kernel::str::CString
                use crate::ast::StringSegment;
                let mut fmt = String::new();
                let mut args: Vec<String> = Vec::new();
                for seg in segs {
                    match seg {
                        StringSegment::Lit(s)  => fmt.push_str(s),
                        StringSegment::Expr(e) => {
                            fmt.push_str("{}");
                            args.push(self.emit_expr(e));
                        }
                        StringSegment::FormattedExpr(e, spec) => {
                            fmt.push_str(&format!("{{:{}}}", spec));
                            args.push(self.emit_expr(e));
                        }
                    }
                }
                if args.is_empty() {
                    format!("c_str!(\"{}\")", escape_str(&fmt))
                } else {
                    // In kernel, format strings use kernel::str::CString::try_from_fmt
                    format!(
                        "kernel::str::CString::try_from_fmt(kernel::fmt!(\"{}\", {})).unwrap()",
                        escape_str(&fmt),
                        args.join(", ")
                    )
                }
            }
            ExprKind::Bool(b)  => b.to_string(),
            ExprKind::Nil      => "None".into(),
            ExprKind::Void     => "()".into(),

            ExprKind::Var(n)   => n.clone(),

            ExprKind::BinOp(op, l, r) => {
                let ls = self.emit_expr(l);
                let rs = self.emit_expr(r);
                format!("({} {} {})", ls, binop_str(op), rs)
            }

            ExprKind::UnaryOp(op, e) => {
                let s = self.emit_expr(e);
                match op {
                    UnaryOp::Neg    => format!("(-{})", s),
                    UnaryOp::Not    => format!("(!{})", s),
                    UnaryOp::BitNot => format!("(!{})", s),
                }
            }

            ExprKind::Assign(lhs, rhs) => {
                let ls = self.emit_expr(lhs);
                let rs = self.emit_expr(rhs);
                format!("{} = {}", ls, rs)
            }

            ExprKind::Field(obj, field) => {
                let obj_s = self.emit_expr(obj);
                format!("{}.{}", obj_s, field)
            }

            ExprKind::Index(obj, idx) => {
                let obj_s = self.emit_expr(obj);
                let idx_s = self.emit_expr(idx);
                format!("{}[{}]", obj_s, idx_s)
            }

            ExprKind::Call(callee, args) => {
                // Special case: print!/println! → kernel::pr_info!
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "print" || name == "println" {
                        return self.emit_print_call(args);
                    }
                }
                let callee_s = self.emit_expr(callee);
                let args_s: Vec<String> = args.iter()
                    .map(|a| self.emit_expr(&a.value))
                    .collect();
                format!("{}({})", callee_s, args_s.join(", "))
            }

            ExprKind::MacroCall { name, args } => {
                // print!/println! → kernel::pr_info!
                if name == "print" || name == "println" {
                    let fake_args: Vec<crate::ast::Arg> = args.iter().map(|a| crate::ast::Arg {
                        label: None,
                        value: a.clone(),
                        spread: false,
                    }).collect();
                    return self.emit_print_call(&fake_args);
                }
                let args_s: Vec<String> = args.iter().map(|a| self.emit_expr(a)).collect();
                format!("{}!({})", name, args_s.join(", "))
            }

            ExprKind::MethodCall(obj, method, args) => {
                let obj_s = self.emit_expr(obj);
                let args_s: Vec<String> = args.iter()
                    .map(|a| self.emit_expr(&a.value))
                    .collect();
                format!("{}.{}({})", obj_s, method, args_s.join(", "))
            }

            ExprKind::Array(elems) => {
                // Vec in kernel
                let elems_s: Vec<String> = elems.iter().map(|e| self.emit_expr(e)).collect();
                format!("kernel::prelude::vec![{}]", elems_s.join(", "))
            }

            ExprKind::Tuple(elems) => {
                let elems_s: Vec<String> = elems.iter().map(|e| self.emit_expr(e)).collect();
                format!("({})", elems_s.join(", "))
            }

            ExprKind::Dict(pairs) => {
                // No convenient macro in kernel; emit an rbtree construction stub
                if pairs.is_empty() {
                    "kernel::rbtree::RBTree::new()".into()
                } else {
                    let mut s = "{ let mut __m = kernel::rbtree::RBTree::new(); ".to_string();
                    for (k, v) in pairs {
                        s.push_str(&format!(
                            "__m.try_insert({}, {}).unwrap(); ",
                            self.emit_expr(k),
                            self.emit_expr(v)
                        ));
                    }
                    s.push_str("__m }");
                    s
                }
            }

            ExprKind::Set(elems) => {
                let mut s = "{ let mut __s = kernel::rbtree::RBTree::new(); ".to_string();
                for e in elems {
                    s.push_str(&format!("__s.try_insert({}, ()).unwrap(); ", self.emit_expr(e)));
                }
                s.push_str("__s }");
                s
            }

            ExprKind::Cast(e, ty) => {
                let es = self.emit_expr(e);
                let ty_s = self.emit_type(ty);
                format!("({} as {})", es, ty_s)
            }

            ExprKind::Range { start, end, inclusive } => {
                let ss = self.emit_expr(start);
                let es = self.emit_expr(end);
                if *inclusive {
                    format!("({}..={})", ss, es)
                } else {
                    format!("({}..{})", ss, es)
                }
            }

            ExprKind::Else(e, default) => {
                // nil-coalescing
                let es = self.emit_expr(e);
                let ds = self.emit_expr(default);
                format!("{}.unwrap_or({})", es, ds)
            }

            ExprKind::If(stmt) => {
                // If as expression — emit as block
                let mut s = String::new();
                for (i, (cond, body)) in stmt.branches.iter().enumerate() {
                    let cond_s = self.emit_expr(cond);
                    if i == 0 {
                        s.push_str(&format!("if {} {{ ", cond_s));
                    } else {
                        s.push_str(&format!(" }} else if {} {{ ", cond_s));
                    }
                    for stmt_b in body {
                        // simple inline — emit as expression
                        s.push_str(&self.emit_stmt_inline(stmt_b));
                    }
                }
                if let Some(else_body) = &stmt.else_body {
                    s.push_str(" } else { ");
                    for stmt_b in else_body {
                        s.push_str(&self.emit_stmt_inline(stmt_b));
                    }
                }
                s.push_str(" }");
                s
            }

            ExprKind::Block(stmts) => {
                let mut s = "{ ".to_string();
                for stmt in stmts {
                    s.push_str(&self.emit_stmt_inline(stmt));
                }
                s.push_str(" }");
                s
            }

            ExprKind::GenericCall(callee, tys, args) => {
                // channel<T> or channel<T, N> → kernel_channel::<T, N>()
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "channel" {
                        let elem_ty = tys.first()
                            .map(|t| self.emit_type(t))
                            .unwrap_or_else(|| "_".into());
                        // Capacity: second type arg (as integer literal) or first call arg,
                        // defaulting to 2.
                        let cap = if tys.len() >= 2 {
                            // channel<T, 32> — capacity stored as a named type constant
                            match &tys[1] {
                                crate::ast::Type::Named(n) => n.clone(),
                                other => self.emit_type(other),
                            }
                        } else if let Some(first_arg) = args.first() {
                            // channel<T>(32) — capacity as call argument
                            self.emit_expr(&first_arg.value)
                        } else {
                            "2".into()
                        };
                        return format!("kernel_channel::<{}, {}>()", elem_ty, cap);
                    }
                }
                let callee_s = self.emit_expr(callee);
                let args_s: Vec<String> = args.iter()
                    .map(|a| self.emit_expr(&a.value))
                    .collect();
                format!("{}({})", callee_s, args_s.join(", "))
            }

            ExprKind::DotIdent(name) => name.clone(),

            // `task: expr` — in kernel context the inner expr is already the wrapper call
            // that returns a KernelFuture<T>.  Just emit it directly.
            ExprKind::Task(inner) => self.emit_expr(inner),

            ExprKind::TaskWithTimeout(_, _) => {
                "/* TODO: kernel task-with-timeout */".into()
            }

            // `join [f1, f2, ...]` — no tokio::join! in the kernel; block sequentially on each
            // KernelFuture via .wait().  The result is a tuple, matching the std backend layout.
            ExprKind::JoinAll(handles) => {
                let parts: Vec<String> = handles.iter()
                    .map(|h| format!("{}.wait()", self.emit_expr(h)))
                    .collect();
                if parts.len() == 1 {
                    parts[0].clone()
                } else {
                    format!("({})", parts.join(", "))
                }
            }
            ExprKind::Closure(_, _, _, _, _) => "/* TODO: kernel closure */".into(),
            ExprKind::Match(_) => "/* TODO: kernel match expr */".into(),
            ExprKind::Do(_) => "/* TODO: kernel do */".into(),
            ExprKind::Loop(_) => "/* TODO: kernel loop expr */".into(),
            ExprKind::TryElse(e, _) => self.emit_expr(e),
            ExprKind::TryElseBlock(_, _) => "/* TODO: kernel try/else */".into(),
            ExprKind::OptionalField(obj, field) => {
                let obj_s = self.emit_expr(obj);
                format!("{}.map(|__v| __v.{})", obj_s, field)
            }
            ExprKind::OptionalMethodCall(obj, method, args) => {
                let obj_s = self.emit_expr(obj);
                let args_s: Vec<String> = args.iter()
                    .map(|a| self.emit_expr(&a.value))
                    .collect();
                format!("{}.map(|__v| __v.{}({}))", obj_s, method, args_s.join(", "))
            }
            ExprKind::Pipe(lhs, fn_name, args) => {
                let lhs_s = self.emit_expr(lhs);
                let args_s: Vec<String> = args.iter()
                    .map(|a| self.emit_expr(&a.value))
                    .collect();
                if args_s.is_empty() {
                    format!("{}({})", fn_name, lhs_s)
                } else {
                    format!("{}({}, {})", fn_name, lhs_s, args_s.join(", "))
                }
            }
        }
    }

    /// Emit a print!/println! call as kernel::pr_info!
    fn emit_print_call(&self, args: &[crate::ast::Arg]) -> String {
        if args.is_empty() {
            return "kernel::pr_info!(\"\\n\")".into();
        }
        // First arg is typically the format string
        let first = &args[0].value;
        match &first.kind {
            ExprKind::Str(s) => {
                if args.len() == 1 {
                    // Ensure there's a newline
                    let msg = escape_str(s);
                    format!("kernel::pr_info!(\"{}\\n\")", msg)
                } else {
                    let rest: Vec<String> = args[1..].iter()
                        .map(|a| self.emit_expr(&a.value))
                        .collect();
                    let msg = escape_str(s);
                    format!("kernel::pr_info!(\"{}\\n\", {})", msg, rest.join(", "))
                }
            }
            _ => {
                let args_s: Vec<String> = args.iter()
                    .map(|a| self.emit_expr(&a.value))
                    .collect();
                format!("kernel::pr_info!({})", args_s.join(", "))
            }
        }
    }

    /// Emit a statement as an inline expression (for use inside expression contexts).
    fn emit_stmt_inline(&self, stmt: &crate::ast::Stmt) -> String {
        match stmt {
            crate::ast::Stmt::Expr(e) => format!("{}; ", self.emit_expr(e)),
            crate::ast::Stmt::Return(r) => {
                if let Some(v) = &r.value {
                    format!("return {}; ", self.emit_expr(v))
                } else {
                    "return; ".into()
                }
            }
            _ => "/* TODO: kernel inline stmt */; ".into(),
        }
    }
}
