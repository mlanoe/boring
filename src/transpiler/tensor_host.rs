//! Rewrites top-level whole-matrix tensor calls into ordinary private kernels.
//! This deliberately reuses the existing kernel constructor/dispatch/readback
//! machinery instead of teaching four host emitters another buffer protocol.
use crate::ast::*;
use std::collections::{HashMap, HashSet};

pub(crate) struct Lowered {
    pub program: Program,
    pub errors: Vec<super::TranspileError>,
}

pub(crate) fn lower(program: &Program) -> Lowered {
    let mut types: HashMap<String, Type> = HashMap::new();
    let mut items = Vec::new();
    let mut kernels = Vec::new();
    let mut errors = Vec::new();
    let mut ordinal = 0usize;
    let mut used_kernel_names: HashSet<String> = program.items.iter().filter_map(|item| match item {
        Item::Kernel(kernel) => Some(kernel.name.clone()),
        _ => None,
    }).collect();
    let mut used_binding_names: HashSet<String> = program.items.iter().filter_map(|item| match item {
        Item::Let(binding) => Some(binding.name.clone()),
        _ => None,
    }).collect();

    for item in &program.items {
        if let Item::Let(binding) = item {
            if let Some(ty) = &binding.ty {
                types.insert(binding.name.clone(), ty.clone());
            }
            items.push(item.clone());
            continue;
        }
        if let Item::Fn(function) = item {
            let mut function = function.clone();
            if !function.throws && function.body.iter().any(stmt_uses_host_tensor) {
                errors.push(super::TranspileError::at_line(
                    "functions containing gpu.tensor host dispatch must declare throws",
                    function.line,
                ));
                items.push(Item::Fn(function));
                continue;
            }
            let mut function_types: HashMap<String, Type> = function.params.iter()
                .filter_map(|param| param.ty.as_ref().map(|ty| (param.name.clone(), ty.clone())))
                .collect();
            for stmt in &function.body {
                if let Stmt::Let(binding) = stmt {
                    used_binding_names.insert(binding.name.clone());
                }
            }
            function.body = lower_function_body(
                &function.body,
                &mut function_types,
                &mut kernels,
                &mut errors,
                &mut ordinal,
                &mut used_kernel_names,
                &mut used_binding_names,
            );
            items.push(Item::Fn(function));
            continue;
        }
        let Item::Stmt(Stmt::Expr(call)) = item else {
            if item_uses_host_tensor(item) {
                errors.push(super::TranspileError::at_line(
                    "gpu.tensor host calls in GPU builds are currently supported only as top-level statements",
                    item_line(item),
                ));
            }
            items.push(item.clone());
            continue;
        };
        let Some((method, names)) = host_call(call) else {
            items.push(item.clone());
            continue;
        };
        let Some(spec) = resolve_spec(method, names, &types) else {
            errors.push(super::TranspileError::at_line(
                "unable to synthesize gpu.tensor host dispatch from the declared operand types",
                call.line,
            ));
            items.push(item.clone());
            continue;
        };
        let (kernel_name, instance) = loop {
            let kernel_name = format!("BoringTensorHost{ordinal}");
            let instance = format!("__boring_tensor_host_{ordinal}");
            ordinal += 1;
            if !used_kernel_names.contains(&kernel_name) && !used_binding_names.contains(&instance) {
                used_kernel_names.insert(kernel_name.clone());
                used_binding_names.insert(instance.clone());
                break (kernel_name, instance);
            }
        };
        match parse_kernel(&kernel_name, &spec, method) {
            Ok(kernel) => kernels.push(Item::Kernel(kernel)),
            Err(message) => {
                errors.push(super::TranspileError::at_line(message, call.line));
                items.push(item.clone());
                continue;
            }
        }
        match parse_replacement(&kernel_name, &instance, names, &spec) {
            Ok(replacement) => items.extend(replacement),
            Err(message) => {
                errors.push(super::TranspileError::at_line(message, call.line));
                items.push(item.clone());
            }
        }
    }
    kernels.extend(items);
    Lowered { program: Program { items: kernels }, errors }
}

fn lower_function_body(
    body: &[Stmt],
    types: &mut HashMap<String, Type>,
    kernels: &mut Vec<Item>,
    errors: &mut Vec<super::TranspileError>,
    ordinal: &mut usize,
    used_kernel_names: &mut HashSet<String>,
    used_binding_names: &mut HashSet<String>,
) -> Vec<Stmt> {
    let mut out = Vec::new();
    let mut index = 0usize;
    while index < body.len() {
        let stmt = &body[index];
        if let Stmt::Let(binding) = stmt {
            if let Some(ty) = &binding.ty {
                types.insert(binding.name.clone(), ty.clone());
            }
            out.push(stmt.clone());
            index += 1;
            continue;
        }
        let Stmt::Expr(call) = stmt else {
            if stmt_uses_host_tensor(stmt) {
                errors.push(super::TranspileError::at_line(
                    "gpu.tensor host calls in GPU builds must be direct function-body statements, outside control flow",
                    stmt_line(stmt),
                ));
            }
            out.push(stmt.clone());
            index += 1;
            continue;
        };
        if let Some(dynamic) = dynamic_linear_call(call) {
            let Some(quals) = resolve_dynamic_quals(dynamic.operands, types) else {
                errors.push(super::TranspileError::at_line(
                    "unable to synthesize dynamic gpu.tensor.linear from the declared operand types",
                    call.line,
                ));
                out.push(stmt.clone());
                index += 1;
                continue;
            };
            let (kernel_name, instance) = allocate_names(ordinal, used_kernel_names, used_binding_names);
            match parse_dynamic_linear_kernel(&kernel_name, quals) {
                Ok(kernel) => kernels.push(Item::Kernel(kernel)),
                Err(message) => {
                    errors.push(super::TranspileError::at_line(message, call.line));
                    out.push(stmt.clone());
                    index += 1;
                    continue;
                }
            }
            match parse_dynamic_replacement_stmts(&kernel_name, &instance, &dynamic) {
                Ok(mut replacement) => {
                    let returns_destination = body.get(index + 1).is_some_and(|next| {
                        matches!(next, Stmt::Expr(expr) if matches!(&expr.kind, ExprKind::Var(name) if name == dynamic.operands[2]))
                    });
                    if returns_destination {
                        replacement.pop();
                        out.extend(replacement);
                        let next = match &body[index + 1] { Stmt::Expr(expr) => expr, _ => unreachable!() };
                        out.push(Stmt::Expr(Expr {
                            kind: ExprKind::Field(Box::new(Expr {
                                kind: ExprKind::Var(instance), line: next.line, col: next.col, len: next.len,
                            }), "c".into()),
                            line: next.line, col: next.col, len: next.len,
                        }));
                        index += 2;
                        continue;
                    }
                    out.extend(replacement);
                }
                Err(message) => {
                    errors.push(super::TranspileError::at_line(message, call.line));
                    out.push(stmt.clone());
                }
            }
            index += 1;
            continue;
        }
        let Some((method, names)) = host_call(call) else {
            out.push(stmt.clone());
            index += 1;
            continue;
        };
        let Some(spec) = resolve_spec(method, names, types) else {
            errors.push(super::TranspileError::at_line(
                "unable to synthesize gpu.tensor host dispatch from the declared operand types",
                call.line,
            ));
            out.push(stmt.clone());
            index += 1;
            continue;
        };
        let (kernel_name, instance) = allocate_names(
            ordinal, used_kernel_names, used_binding_names,
        );
        match parse_kernel(&kernel_name, &spec, method) {
            Ok(kernel) => kernels.push(Item::Kernel(kernel)),
            Err(message) => {
                errors.push(super::TranspileError::at_line(message, call.line));
                out.push(stmt.clone());
                index += 1;
                continue;
            }
        }
        match parse_replacement_stmts(&kernel_name, &instance, names, &spec) {
            Ok(mut replacement) => {
                let returns_destination = body.get(index + 1).is_some_and(|next| {
                    matches!(next, Stmt::Expr(expr) if matches!(&expr.kind, ExprKind::Var(name) if name == names[2]))
                });
                if returns_destination {
                    replacement.pop(); // Keep the kernel field resident until the function return.
                    out.extend(replacement);
                    let next = match &body[index + 1] { Stmt::Expr(expr) => expr, _ => unreachable!() };
                    out.push(Stmt::Expr(Expr {
                        kind: ExprKind::Field(Box::new(Expr {
                            kind: ExprKind::Var(instance.clone()),
                            line: next.line, col: next.col, len: next.len,
                        }), "c".into()),
                        line: next.line, col: next.col, len: next.len,
                    }));
                    index += 2;
                    continue;
                }
                out.extend(replacement);
            }
            Err(message) => {
                errors.push(super::TranspileError::at_line(message, call.line));
                out.push(stmt.clone());
            }
        }
        index += 1;
    }
    out
}

struct DynamicLinearCall<'a> {
    operands: [&'a str; 3],
    m: String,
    n: String,
    k: String,
}

fn dimension_source(expr: &Expr) -> Option<String> {
    match &expr.kind {
        ExprKind::Var(name) => Some(name.clone()),
        ExprKind::Int(value) if *value > 0 => Some(value.to_string()),
        _ => None,
    }
}

fn dynamic_linear_call(expr: &Expr) -> Option<DynamicLinearCall<'_>> {
    let ExprKind::MethodCall(receiver, method, args) = &expr.kind else { return None };
    if method != "linear" || args.len() != 6 { return None; }
    if !matches!(&receiver.kind, ExprKind::Field(gpu, ns)
        if ns == "tensor" && matches!(&gpu.kind, ExprKind::Var(name) if name == "gpu")) { return None; }
    let operands: Vec<&str> = args[..3].iter().map(|arg| match &arg.value.kind {
        ExprKind::Var(name) if arg.label.is_none() => Some(name.as_str()),
        _ => None,
    }).collect::<Option<_>>()?;
    let dim = |label: &str| args[3..].iter().find(|arg| arg.label.as_deref() == Some(label))
        .and_then(|arg| dimension_source(&arg.value));
    Some(DynamicLinearCall { operands: operands.try_into().ok()?, m: dim("m")?, n: dim("n")?, k: dim("k")? })
}

fn resolve_dynamic_quals(names: [&str; 3], types: &HashMap<String, Type>) -> Option<[GpuQual; 3]> {
    names.map(|name| {
        let mut ty = types.get(name)?;
        while let Type::Mut(inner) = ty { ty = inner; }
        let Type::Qualified(_, owner) = ty else { return None };
        match owner {
            OwnerQual::GpuGlobal => Some(GpuQual::Global),
            OwnerQual::GpuUnified => Some(GpuQual::Unified),
            _ => None,
        }
    }).into_iter().collect::<Option<Vec<_>>>()?.try_into().ok()
}

fn allocate_names(
    ordinal: &mut usize,
    used_kernel_names: &mut HashSet<String>,
    used_binding_names: &mut HashSet<String>,
) -> (String, String) {
    loop {
        let kernel_name = format!("BoringTensorHost{}", *ordinal);
        let instance = format!("__boring_tensor_host_{}", *ordinal);
        *ordinal += 1;
        if !used_kernel_names.contains(&kernel_name) && !used_binding_names.contains(&instance) {
            used_kernel_names.insert(kernel_name.clone());
            used_binding_names.insert(instance.clone());
            return (kernel_name, instance);
        }
    }
}

#[derive(Clone)]
struct Spec {
    m: usize,
    n: usize,
    k: usize,
    quals: [GpuQual; 3],
    transpose_b: bool,
}

fn resolve_spec(method: &str, names: [&str; 3], types: &HashMap<String, Type>) -> Option<Spec> {
    if !matches!(method, "matmul" | "mma" | "linear") { return None; }
    let mut shapes = Vec::new();
    let mut quals = Vec::new();
    for name in names {
        let mut ty = types.get(name)?;
        while let Type::Mut(inner) = ty { ty = inner; }
        let Type::Qualified(inner, owner) = ty else { return None };
        quals.push(match owner {
            OwnerQual::GpuGlobal => GpuQual::Global,
            OwnerQual::GpuUnified => GpuQual::Unified,
            _ => return None,
        });
        let Type::LabeledArray(_, axes) = inner.without_mut() else { return None };
        if axes.len() != 2 { return None; }
        let extent = |i: usize| match &axes[i].size.as_ref()?.0.kind {
            ExprKind::Int(value) => usize::try_from(*value).ok(),
            _ => None,
        };
        shapes.push([extent(0)?, extent(1)?]);
    }
    let (k, m) = (shapes[0][0], shapes[0][1]);
    let transpose_b = method == "linear";
    let n = if transpose_b { shapes[1][1] } else { shapes[1][0] };
    let right_k = if transpose_b { shapes[1][0] } else { shapes[1][1] };
    if right_k != k || shapes[2] != [n, m] { return None; }
    Some(Spec { m, n, k, quals: quals.try_into().ok()?, transpose_b })
}

fn qual_source(qual: &GpuQual) -> &'static str {
    match qual {
        GpuQual::Global => "global",
        GpuQual::Unified => "unified",
        _ => unreachable!(),
    }
}

fn parse_kernel(name: &str, spec: &Spec, method: &str) -> Result<KernelDecl, String> {
    let tile_rows = spec.m.min(16).max(1);
    let tile_cols = spec.n.min(16).max(1);
    let tile_method = match method {
        "mma" => "mmaTile",
        "linear" => "linearTile",
        _ => "matmulTile",
    };
    let b_axes = if spec.transpose_b {
        format!("k = {}, n = {}", spec.k, spec.n)
    } else {
        format!("n = {}, k = {}", spec.n, spec.k)
    };
    let source = format!(
        "kernel {name}:\n    let [float32, k = {k}, m = {m}]'{qa} a\n    let [float32, {b_axes}]'{qb} b\n    mut [float32, n = {n}, m = {m}]'{qc} c\n    init([float32, k = {k}, m = {m}]'{qa} input_a, [float32, {b_axes}]'{qb} input_b, [float32, n = {n}, m = {m}]'{qc} input_c):\n        a = input_a\n        b = input_b\n        c = input_c\n    def ():\n        let row = gpu.block.y * {tile_rows}\n        let col = gpu.block.x * {tile_cols}\n        gpu.tensor.{tile_method}(a, b, c, row = row, col = col, rows = {tile_rows}, cols = {tile_cols})\n",
        k = spec.k, m = spec.m, n = spec.n,
        qa = qual_source(&spec.quals[0]), qb = qual_source(&spec.quals[1]), qc = qual_source(&spec.quals[2]),
    );
    let parsed = crate::parser::parse(crate::lexer::lex(&source).map_err(|e| format!("tensor kernel lex error: {e:?}"))?)
        .map_err(|e| format!("tensor kernel parse error: {}", e.msg()))?;
    match parsed.items.into_iter().next() {
        Some(Item::Kernel(kernel)) => Ok(kernel),
        _ => Err("tensor kernel synthesis produced no kernel".into()),
    }
}

fn parse_dynamic_linear_kernel(name: &str, quals: [GpuQual; 3]) -> Result<KernelDecl, String> {
    let source = format!(
        "kernel {name}:\n    let [float32]'{qa} a\n    let [float32]'{qb} b\n    mut [float32]'{qc} c\n    let int m\n    let int n\n    let int k\n    init([float32]'{qa} input_a, [float32]'{qb} input_b, [float32]'{qc} input_c, int input_m, int input_n, int input_k):\n        a = input_a\n        b = input_b\n        c = input_c\n        m = input_m\n        n = input_n\n        k = input_k\n    def ():\n        let blockIndex = gpu.block.x + gpu.block.y * gpu.gridDim.x\n        let cell = gpu.thread.x + blockIndex * gpu.blockDim.x\n        if cell < m * n:\n            let row = cell / n\n            let col = cell % n\n            var float32 sum = 0.0\n            for inner in 0..<k:\n                sum += a[row * k + inner] * b[col * k + inner]\n            c[row * n + col] = sum\n",
        qa = qual_source(&quals[0]), qb = qual_source(&quals[1]), qc = qual_source(&quals[2]),
    );
    let parsed = crate::parser::parse(
        crate::lexer::lex(&source).map_err(|e| format!("dynamic tensor kernel lex error: {e:?}"))?
    ).map_err(|e| format!("dynamic tensor kernel parse error: {}", e.msg()))?;
    match parsed.items.into_iter().next() {
        Some(Item::Kernel(kernel)) => Ok(kernel),
        _ => Err("dynamic tensor kernel synthesis produced no kernel".into()),
    }
}

fn parse_replacement(name: &str, instance: &str, operands: [&str; 3], spec: &Spec) -> Result<Vec<Item>, String> {
    let tile_rows = spec.m.min(16).max(1);
    let tile_cols = spec.n.min(16).max(1);
    let gx = spec.n.div_ceil(tile_cols);
    let gy = spec.m.div_ceil(tile_rows);
    let source = format!(
        "mut {instance} = {name}({a}, {b}, {c})\nkernel:\n    {instance}(block = 256, grid = ({gx}, {gy}))\n{c} = {instance}.c\n",
        a = operands[0], b = operands[1], c = operands[2],
    );
    crate::parser::parse(crate::lexer::lex(&source).map_err(|e| format!("tensor dispatch lex error: {e:?}"))?)
        .map(|program| program.items)
        .map_err(|e| format!("tensor dispatch parse error: {}", e.msg()))
}

fn parse_replacement_stmts(name: &str, instance: &str, operands: [&str; 3], spec: &Spec) -> Result<Vec<Stmt>, String> {
    let tile_rows = spec.m.min(16).max(1);
    let tile_cols = spec.n.min(16).max(1);
    let gx = spec.n.div_ceil(tile_cols);
    let gy = spec.m.div_ceil(tile_rows);
    let source = format!(
        "def __tensor_wrapper():\n    mut {instance} = {name}({a}, {b}, {c})\n    kernel:\n        {instance}(block = 256, grid = ({gx}, {gy}))\n    {c} = {instance}.c\n",
        a = operands[0], b = operands[1], c = operands[2],
    );
    let parsed = crate::parser::parse(
        crate::lexer::lex(&source).map_err(|e| format!("tensor dispatch lex error: {e:?}"))?
    ).map_err(|e| format!("tensor dispatch parse error: {}", e.msg()))?;
    match parsed.items.into_iter().next() {
        Some(Item::Fn(function)) => Ok(function.body),
        _ => Err("tensor dispatch synthesis produced no function body".into()),
    }
}

fn parse_dynamic_replacement_stmts(
    name: &str,
    instance: &str,
    call: &DynamicLinearCall<'_>,
) -> Result<Vec<Stmt>, String> {
    let source = format!(
        "def __tensor_wrapper() throws:\n    guard {m} > 0 else throw \"tensor dimension m must be positive\"\n    guard {n} > 0 else throw \"tensor dimension n must be positive\"\n    guard {k} > 0 else throw \"tensor dimension k must be positive\"\n    guard {a}.length == {m} * {k} else throw \"tensor left operand length mismatch\"\n    guard {b}.length == {n} * {k} else throw \"tensor weight length mismatch\"\n    guard {c}.length == {m} * {n} else throw \"tensor destination length mismatch\"\n    mut {instance} = {name}({a}, {b}, {c}, {m}, {n}, {k})\n    kernel:\n        {instance}(block = 256, grid = ({m} * {n} + 255) / 256)\n    {c} = {instance}.c\n",
        a = call.operands[0], b = call.operands[1], c = call.operands[2],
        m = call.m, n = call.n, k = call.k,
    );
    let parsed = crate::parser::parse(
        crate::lexer::lex(&source).map_err(|e| format!("dynamic tensor dispatch lex error: {e:?}"))?
    ).map_err(|e| format!("dynamic tensor dispatch parse error: {}", e.msg()))?;
    match parsed.items.into_iter().next() {
        Some(Item::Fn(function)) => Ok(function.body),
        _ => Err("dynamic tensor dispatch synthesis produced no function body".into()),
    }
}

fn host_call(expr: &Expr) -> Option<(&str, [&str; 3])> {
    let ExprKind::MethodCall(receiver, method, args) = &expr.kind else { return None };
    if !matches!(method.as_str(), "matmul" | "mma" | "linear") || args.len() != 3 { return None; }
    if !matches!(&receiver.kind, ExprKind::Field(gpu, ns)
        if ns == "tensor" && matches!(&gpu.kind, ExprKind::Var(name) if name == "gpu")) { return None; }
    let names: Option<Vec<&str>> = args.iter().map(|arg| match &arg.value.kind {
        ExprKind::Var(name) => Some(name.as_str()),
        _ => None,
    }).collect();
    names?.try_into().ok().map(|names| (method.as_str(), names))
}

fn item_uses_host_tensor(item: &Item) -> bool {
    match item {
        Item::Fn(function) => function.body.iter().any(stmt_uses_host_tensor),
        Item::Stmt(stmt) => stmt_uses_host_tensor(stmt),
        _ => false,
    }
}

fn stmt_uses_host_tensor(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Expr(expr) => host_call(expr).is_some() || dynamic_linear_call(expr).is_some(),
        Stmt::If(value) => value.branches.iter().any(|(_, body)| body.iter().any(stmt_uses_host_tensor))
            || value.else_body.as_ref().is_some_and(|body| body.iter().any(stmt_uses_host_tensor)),
        Stmt::While(value) => value.body.iter().any(stmt_uses_host_tensor),
        Stmt::For(value) => value.body.iter().any(stmt_uses_host_tensor),
        Stmt::Loop(value) => value.body.iter().any(stmt_uses_host_tensor),
        Stmt::DoWhile(value) => value.body.iter().any(stmt_uses_host_tensor),
        Stmt::KernelBlock(value) => value.body.iter().any(stmt_uses_host_tensor),
        _ => false,
    }
}

fn item_line(item: &Item) -> usize {
    match item {
        Item::Fn(value) => value.line,
        Item::Stmt(Stmt::Expr(value)) => value.line,
        _ => 0,
    }
}

fn stmt_line(stmt: &Stmt) -> usize {
    match stmt {
        Stmt::Expr(value) => value.line,
        Stmt::Let(value) => value.line,
        Stmt::If(value) => value.line,
        Stmt::While(value) => value.line,
        Stmt::For(value) => value.line,
        Stmt::Loop(value) => value.line,
        Stmt::DoWhile(value) => value.line,
        Stmt::KernelBlock(value) => value.line,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> Program {
        crate::parser::parse(crate::lexer::lex(source).unwrap()).unwrap()
    }

    #[test]
    fn lowers_top_level_calls_to_private_kernels_and_dispatches() {
        let program = parse("let [float32, k = 17, m = 33]'gpu'global a = [0.0 for ..<561]\nlet [float32, n = 35, k = 17]'gpu'unified b = [0.0 for ..<595]\nmut [float32, n = 35, m = 33]'gpu'unified c = [0.0 for ..<1155]\ngpu.tensor.matmul(a, b, c)\ngpu.tensor.mma(a, b, c)\n");
        let lowered = lower(&program);
        assert!(lowered.errors.is_empty(), "{:?}", lowered.errors);
        let kernels: Vec<_> = lowered.program.items.iter().filter_map(|item| match item {
            Item::Kernel(kernel) => Some(kernel.name.as_str()),
            _ => None,
        }).collect();
        assert_eq!(kernels, ["BoringTensorHost0", "BoringTensorHost1"]);
        assert_eq!(lowered.program.items.iter().filter(|item| matches!(item, Item::Stmt(Stmt::KernelBlock(_)))).count(), 2);
    }

    #[test]
    fn lowers_linear_with_native_row_major_weights() {
        let program = parse("let [float32, k = 5, m = 3]'gpu'global x = [0.0 for ..<15]\nlet [float32, k = 5, n = 7]'gpu'global w = [0.0 for ..<35]\nmut [float32, n = 7, m = 3]'gpu'unified y = [0.0 for ..<21]\ngpu.tensor.linear(x, w, y)\n");
        let lowered = lower(&program);
        assert!(lowered.errors.is_empty(), "{:?}", lowered.errors);
        let kernel = lowered.program.items.iter().find_map(|item| match item {
            Item::Kernel(kernel) if kernel.name == "BoringTensorHost0" => Some(kernel),
            _ => None,
        }).unwrap();
        let Stmt::Expr(call) = &kernel.methods[0].body[2] else { panic!("tensor call") };
        let operation = crate::checker::tensor::resolve(call, kernel).unwrap();
        assert!(operation.transpose_b);
    }

    #[test]
    fn rejects_nested_host_scheduling() {
        let lowered = lower(&parse("def work() throws:\n    let [float32, k = 2, m = 2]'gpu'global a = [0.0 for ..<4]\n    let [float32, n = 2, k = 2]'gpu'global b = [0.0 for ..<4]\n    mut [float32, n = 2, m = 2]'gpu'unified c = [0.0 for ..<4]\n    if true:\n        gpu.tensor.matmul(a, b, c)\n"));
        assert!(lowered.errors.iter().any(|error| error.message.contains("outside control flow")));
    }

    #[test]
    fn lowers_direct_calls_inside_req_functions() {
        let lowered = lower(&parse("req [float32]'gpu'unified compute([float32, k = 3, m = 2]'gpu'global x, [float32, k = 3, n = 2]'gpu'global w) throws:\n    mut [float32, n = 2, m = 2]'gpu'unified y = [0.0 for ..<4]\n    gpu.tensor.linear(x, w, y)\n    y\n"));
        assert!(lowered.errors.is_empty(), "{:?}", lowered.errors);
        assert!(lowered.program.items.iter().any(|item| matches!(item, Item::Kernel(kernel) if kernel.name == "BoringTensorHost0")));
        let function = lowered.program.items.iter().find_map(|item| match item {
            Item::Fn(function) if function.name == "compute" => Some(function),
            _ => None,
        }).unwrap();
        assert!(function.body.iter().any(|stmt| matches!(stmt, Stmt::KernelBlock(_))));
    }

    #[test]
    fn lowers_dynamic_linear_inside_req_and_returns_resident_field() {
        let source = "req [float32]'gpu'unified compute([float32]'gpu'global x, [float32]'gpu'global w, int m, int n, int k) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32 for ..<m * n]\n    gpu.tensor.linear(x, w, y, m = m, n = n, k = k)\n    y\n";
        let lowered = lower(&parse(source));
        assert!(lowered.errors.is_empty(), "{:?}", lowered.errors);
        let kernel = lowered.program.items.iter().find_map(|item| match item {
            Item::Kernel(kernel) if kernel.name == "BoringTensorHost0" => Some(kernel),
            _ => None,
        }).expect("dynamic kernel");
        assert_eq!(kernel.fields.len(), 6);
        let function = lowered.program.items.iter().find_map(|item| match item {
            Item::Fn(function) if function.name == "compute" => Some(function),
            _ => None,
        }).unwrap();
        assert!(function.body.iter().any(|stmt| matches!(stmt, Stmt::KernelBlock(_))));
        assert!(matches!(function.body.last(), Some(Stmt::Expr(Expr { kind: ExprKind::Field(_, field), .. })) if field == "c"));
    }

    #[test]
    fn synthesized_names_do_not_collide_with_user_names() {
        let program = parse("kernel BoringTensorHost0:\n    def ():\n        let x = 0\nlet __boring_tensor_host_1 = 0\nlet [float32, k = 2, m = 2]'gpu'global a = [0.0 for ..<4]\nlet [float32, n = 2, k = 2]'gpu'global b = [0.0 for ..<4]\nmut [float32, n = 2, m = 2]'gpu'unified c = [0.0 for ..<4]\ngpu.tensor.matmul(a, b, c)\n");
        let lowered = lower(&program);
        assert!(lowered.errors.is_empty(), "{:?}", lowered.errors);
        assert!(lowered.program.items.iter().any(|item| matches!(item, Item::Kernel(kernel) if kernel.name == "BoringTensorHost2")));
    }
}
