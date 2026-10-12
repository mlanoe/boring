//! Lowers `kernel(block = ..., grid = ...): ...` expressions to the existing
//! named-kernel representation.  Keeping this as an early AST pass means the
//! interpreter and every GPU backend share exactly the same implementation.

use std::collections::{HashMap, HashSet};

use crate::ast::*;

pub fn desugar_inline_kernels(program: Program) -> Program {
    let mut generated = Vec::new();
    let mut items = Vec::with_capacity(program.items.len());
    let mut serial = 0usize;

    for mut item in program.items {
        if let Item::Fn(function) = &mut item {
            lower_function(function, &mut generated, &mut serial);
        }
        items.push(item);
    }
    // Declarations may appear after their use in Boring. Appending also keeps
    // imports and the user's source order intact for diagnostics.
    items.extend(generated.into_iter().map(Item::Kernel));
    Program { items }
}

fn lower_function(function: &mut FnDecl, generated: &mut Vec<KernelDecl>, serial: &mut usize) {
    let mut scope = HashMap::<String, Type>::new();
    for param in &function.params {
        if let Some(ty) = &param.ty {
            scope.insert(param.name.clone(), ty.clone());
        }
    }

    let mut lowered_body = Vec::with_capacity(function.body.len());
    for stmt in std::mem::take(&mut function.body) {
        match stmt {
            Stmt::Let(mut binding) => {
                let mut prefix = Vec::new();
                if let Some(value) = &mut binding.value {
                    if lower_expr(value, &scope, generated, serial) {
                        if let ExprKind::Block(mut statements) = std::mem::replace(&mut value.kind, ExprKind::Void) {
                            if let Some(Stmt::Expr(tail)) = statements.pop() {
                                prefix = statements;
                                *value = tail;
                            }
                        }
                    }
                }
                if let Some(ty) = &binding.ty {
                    scope.insert(binding.name.clone(), ty.clone());
                }
                lowered_body.extend(prefix);
                lowered_body.push(Stmt::Let(binding));
            }
            Stmt::Return(mut ret) => {
                let mut prefix = Vec::new();
                if let Some(value) = &mut ret.value {
                    if lower_expr(value, &scope, generated, serial) {
                        if let ExprKind::Block(mut statements) = std::mem::replace(&mut value.kind, ExprKind::Void) {
                            if let Some(Stmt::Expr(tail)) = statements.pop() {
                                prefix = statements;
                                *value = tail;
                            }
                        }
                    }
                }
                lowered_body.extend(prefix);
                lowered_body.push(Stmt::Return(ret));
            }
            Stmt::Expr(mut expression) => {
                if lower_expr(&mut expression, &scope, generated, serial) {
                    if let ExprKind::Block(statements) = expression.kind {
                        lowered_body.extend(statements);
                    } else {
                        lowered_body.push(Stmt::Expr(expression));
                    }
                } else {
                    lowered_body.push(Stmt::Expr(expression));
                }
            }
            _ => lowered_body.push(stmt),
        }
    }
    function.body = lowered_body;
}

fn lower_expr(
    expr: &mut Expr,
    scope: &HashMap<String, Type>,
    generated: &mut Vec<KernelDecl>,
    serial: &mut usize,
) -> bool {
    let ExprKind::KernelLaunch { config, kernel } = &expr.kind else { return false };
    let ExprKind::Block(body) = &kernel.kind else { return false };
    let mut body = body.clone();
    let Some(Stmt::Expr(result)) = body.pop() else {
        return false; // ordinary `kernel(...) value` launch, not the inline-block form
    };

    let output_names: Vec<String> = match &result.kind {
        ExprKind::Var(name) => vec![name.clone()],
        ExprKind::Tuple(values) => values.iter().filter_map(|value| {
            if let ExprKind::Var(name) = &value.kind { Some(name.clone()) } else { None }
        }).collect(),
        _ => return false,
    };
    if output_names.is_empty() { return false; }

    let output_set: HashSet<&str> = output_names.iter().map(String::as_str).collect();
    let mut output_decls = Vec::<LetStmt>::new();
    let mut device_body = Vec::new();
    let mut local_names = HashSet::<String>::new();
    for stmt in body {
        if let Stmt::Let(binding) = &stmt {
            local_names.insert(binding.name.clone());
            if output_set.contains(binding.name.as_str()) {
                output_decls.push(binding.clone());
                continue;
            }
        }
        device_body.push(stmt);
    }
    if output_decls.len() != output_names.len() { return false; }

    let mut used = HashSet::<String>::new();
    crate::ast::walk_stmts(&device_body, &mut |node| {
        if let WalkNode::Expr(Expr { kind: ExprKind::Var(name), .. }) = node {
            used.insert(name.clone());
        }
    });
    // Allocation expressions execute in the generated constructor, but they
    // capture the same enclosing scope as the device body (`n` in
    // `[0.0 for ..<n]` is the common case).
    for output in &output_decls {
        crate::ast::walk_stmts(&[Stmt::Let(output.clone())], &mut |node| {
            if let WalkNode::Expr(Expr { kind: ExprKind::Var(name), .. }) = node {
                used.insert(name.clone());
            }
        });
    }

    let mut captures: Vec<(String, Type)> = scope.iter()
        .filter(|(name, _)| used.contains(*name) && !local_names.contains(*name))
        .map(|(name, ty)| (name.clone(), ty.clone()))
        .collect();
    captures.sort_by(|a, b| a.0.cmp(&b.0));

    *serial += 1;
    let kernel_name = format!("__InlineKernel_{}_{}", expr.line, serial);
    let instance_name = format!("__inline_kernel_{}_{}", expr.line, serial);

    let mut fields = Vec::new();
    let mut init_params = Vec::new();
    let mut init_body = Vec::new();

    for (name, ty) in &captures {
        let (qual, base) = kernel_field_type(ty, FieldBinding::Let);
        fields.push(KernelFieldDecl {
            name: name.clone(), binding: FieldBinding::Let, qual, ty: base,
            default: None, line: expr.line, col: expr.col,
        });
        let param_name = format!("__capture_{name}");
        init_params.push(InitParam {
            is_pub: false, mutable: false, name: param_name.clone(),
            ty: Some(strip_mut(ty).clone()), default: None, line: expr.line, col: expr.col,
        });
        init_body.push(assign_stmt(name, var_expr(&param_name, expr.line, expr.col), expr.line, expr.col));
    }

    for output in &output_decls {
        let Some(ty) = output.ty.as_ref() else { return false };
        let field_binding = match output.binding {
            BindingKind::Let => FieldBinding::Let,
            BindingKind::Mut => FieldBinding::Mut,
            BindingKind::Var => FieldBinding::Var,
            BindingKind::Lazy => return false,
        };
        let (qual, base) = kernel_field_type(ty, field_binding.clone());
        fields.push(KernelFieldDecl {
            name: output.name.clone(), binding: field_binding, qual, ty: base,
            default: None, line: output.line, col: output.col,
        });
        let Some(initializer) = output.value.clone() else { return false };
        init_body.push(assign_stmt(&output.name, initializer, output.line, output.col));
    }

    let entry = FnDecl {
        name: String::new(), qualifier: None, params: vec![], return_ty: None,
        body: device_body, is_pub: false, throws: false, task: false, stream: false,
        stream_capacity: None, mutating: true, return_mutable: false, is_native: false,
        throws_ty: None, type_params: vec![], where_clause: vec![], attrs: vec![],
        line: expr.line, col: expr.col,
    };
    generated.push(KernelDecl {
        name: kernel_name.clone(), is_pub: false, fields,
        inits: vec![InitDecl { params: init_params, body: init_body, line: expr.line, col: expr.col }],
        methods: vec![entry], type_params: vec![], where_clause: vec![], line: expr.line, col: expr.col,
    });

    let ctor = Expr {
        kind: ExprKind::Call(
            Box::new(var_expr(&kernel_name, expr.line, expr.col)),
            captures.iter().map(|(name, _)| Arg {
                label: None, value: var_expr(name, expr.line, expr.col), spread: false, default_rest: false,
            }).collect(),
        ), line: expr.line, col: expr.col, len: expr.len,
    };
    // An inline kernel has no user-visible constructor, so its placement is
    // transferred to the generated private kernel instance before launch.
    let ctor = if let Some(device) = &config.device {
        Expr {
            kind: ExprKind::New { arena: Some(Box::new(device.clone())), ctor: Box::new(ctor) },
            line: expr.line, col: expr.col, len: expr.len,
        }
    } else {
        ctor
    };
    let instance = LetStmt {
        binding: BindingKind::Mut, is_pub: false, is_static: false,
        name: instance_name.clone(), ty: None, var_mut: false, value: Some(ctor),
        is_lazy: false, line: expr.line, col: expr.col,
    };
    let mut launch_args = Vec::new();
    if let Some(value) = &config.block { launch_args.push(labeled_arg("block", value.clone())); }
    if let Some(value) = &config.grid { launch_args.push(labeled_arg("grid", value.clone())); }
    if let Some(value) = &config.after { launch_args.push(labeled_arg("after", value.clone())); }
    if let Some(value) = &config.priority {
        launch_args.push(labeled_arg("priority", var_expr(value, expr.line, expr.col)));
    }
    let launch = Stmt::KernelBlock(KernelBlockStmt {
        body: vec![Stmt::Expr(Expr {
            kind: ExprKind::Call(Box::new(var_expr(&instance_name, expr.line, expr.col)), launch_args),
            line: expr.line, col: expr.col, len: expr.len,
        })], line: expr.line, col: expr.col,
    });
    let result = if output_names.len() == 1 {
        field_expr(&instance_name, &output_names[0], expr.line, expr.col)
    } else {
        Expr {
            kind: ExprKind::Tuple(output_names.iter().map(|name| field_expr(&instance_name, name, expr.line, expr.col)).collect()),
            line: expr.line, col: expr.col, len: expr.len,
        }
    };
    expr.kind = ExprKind::Block(vec![Stmt::Let(instance), launch, Stmt::Expr(result)]);
    true
}

fn strip_mut(ty: &Type) -> &Type {
    if let Type::Mut(inner) = ty { strip_mut(inner) } else { ty }
}

fn kernel_field_type(ty: &Type, binding: FieldBinding) -> (GpuQual, Type) {
    let ty = strip_mut(ty);
    match ty {
        Type::Qualified(inner, OwnerQual::GpuUnified) => (GpuQual::Unified, (**inner).clone()),
        Type::Qualified(inner, OwnerQual::GpuGlobal) => (GpuQual::Global, (**inner).clone()),
        Type::Qualified(inner, OwnerQual::GpuLocal) => (GpuQual::Local, (**inner).clone()),
        Type::Qualified(inner, OwnerQual::GpuConst) => (GpuQual::Const, (**inner).clone()),
        Type::Qualified(inner, OwnerQual::GpuActorGlobal) => (GpuQual::ActorGlobal, (**inner).clone()),
        Type::Qualified(inner, OwnerQual::GpuActorUnified) => (GpuQual::ActorUnified, (**inner).clone()),
        Type::Qualified(inner, OwnerQual::GpuSurface) => (GpuQual::Surface, (**inner).clone()),
        Type::Qualified(inner, OwnerQual::Actor) => (GpuQual::Actor, (**inner).clone()),
        _ => {
            let qual = if matches!(binding, FieldBinding::Let) { GpuQual::Const } else { GpuQual::Local };
            (qual, ty.clone())
        }
    }
}

fn var_expr(name: &str, line: usize, col: usize) -> Expr {
    Expr { kind: ExprKind::Var(name.to_string()), line, col, len: name.len() }
}

fn field_expr(instance: &str, field: &str, line: usize, col: usize) -> Expr {
    Expr {
        kind: ExprKind::Field(Box::new(var_expr(instance, line, col)), field.to_string()),
        line, col, len: instance.len() + field.len() + 1,
    }
}

fn assign_stmt(name: &str, value: Expr, line: usize, col: usize) -> Stmt {
    Stmt::Expr(Expr {
        kind: ExprKind::Assign(Box::new(var_expr(name, line, col)), Box::new(value)),
        line, col, len: name.len(),
    })
}

fn labeled_arg(label: &str, value: Expr) -> Arg {
    Arg { label: Some(label.to_string()), value, spread: false, default_rest: false }
}
