//! Rewrites top-level whole-matrix tensor calls into ordinary private kernels.
//! This deliberately reuses the existing kernel constructor/dispatch/readback
//! machinery instead of teaching four host emitters another buffer protocol.
use crate::ast::*;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub(crate) struct TensorLinearConfig {
    pub matrix_algorithm: Option<String>,
    pub decode_algorithm: Option<String>,
    pub prefill_algorithm: Option<String>,
    pub decode_formats: HashMap<String, String>,
    pub prefill_formats: HashMap<String, String>,
    pub target_warp_width: usize,
    pub native_fixed_matrices: bool,
}

impl Default for TensorLinearConfig {
    fn default() -> Self {
        Self {
            matrix_algorithm: None,
            decode_algorithm: None,
            prefill_algorithm: None,
            decode_formats: HashMap::new(),
            prefill_formats: HashMap::new(),
            target_warp_width: 32,
            native_fixed_matrices: false,
        }
    }
}

fn fixed_geometry(spec: &Spec, config: &TensorLinearConfig) -> (usize, usize, usize) {
    if config.native_fixed_matrices && spec.m % 8 == 0 && spec.n % 8 == 0 && spec.k % 8 == 0 {
        (8, 8, 32)
    } else {
        (spec.m.clamp(1, 16), spec.n.clamp(1, 16), 256)
    }
}

impl TensorLinearConfig {
    fn algorithm(&self, format: Option<&str>, decode: bool) -> &str {
        let formats = if decode { &self.decode_formats } else { &self.prefill_formats };
        let fallback = if decode { &self.decode_algorithm } else { &self.prefill_algorithm };
        format.and_then(|name| formats.get(name)).or(fallback.as_ref()).map_or("auto", String::as_str)
    }
}

pub(crate) struct Lowered {
    pub program: Program,
    pub errors: Vec<super::TranspileError>,
}

#[allow(dead_code)]
pub(crate) fn lower(program: &Program) -> Lowered {
    lower_with_config(program, &TensorLinearConfig::default())
}

pub(crate) fn lower_with_config(program: &Program, tensor_config: &TensorLinearConfig) -> Lowered {
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
                tensor_config,
            );
            items.push(Item::Fn(function));
            continue;
        }
        if let Item::Stmt(stmt) = item {
            if !matches!(stmt, Stmt::Expr(_)) {
                // Function bodies already lower a `for`/`if`/`while`/... wrapping a
                // host tensor call via this exact helper (see `lower_function_body`
                // above) -- top-level control flow around the same call used to hit
                // the generic "only as top-level statements" error below instead,
                // forcing any repeated/conditional dispatch (e.g. a benchmark loop)
                // to be wrapped in a throwaway function just to get the looping this
                // already supports one scope down. Trying the same lowering here
                // first closes that gap; `lower_control_flow_stmt` still returns
                // `None` (falling through to the error) for any statement kind it
                // doesn't specifically handle.
                if let Some(lowered) = lower_control_flow_stmt(
                    stmt, &types, &mut kernels, &mut errors, &mut ordinal,
                    &mut used_kernel_names, &mut used_binding_names, tensor_config,
                ) {
                    items.push(Item::Stmt(lowered));
                    continue;
                }
            }
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
        match parse_kernel(&kernel_name, &spec, method, tensor_config) {
            Ok(kernel) => kernels.push(Item::Kernel(kernel)),
            Err(message) => {
                errors.push(super::TranspileError::at_line(message, call.line));
                items.push(item.clone());
                continue;
            }
        }
        match parse_replacement(&kernel_name, &instance, names, &spec, tensor_config) {
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

#[allow(clippy::too_many_arguments)]
fn lower_function_body(
    body: &[Stmt],
    types: &mut HashMap<String, Type>,
    kernels: &mut Vec<Item>,
    errors: &mut Vec<super::TranspileError>,
    ordinal: &mut usize,
    used_kernel_names: &mut HashSet<String>,
    used_binding_names: &mut HashSet<String>,
    tensor_config: &TensorLinearConfig,
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
        if let Some(lowered) = lower_control_flow_stmt(
            stmt, types, kernels, errors, ordinal, used_kernel_names, used_binding_names, tensor_config,
        ) {
            out.push(lowered);
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
            match parse_dynamic_linear_kernel(&kernel_name, quals, dynamic.bias.is_some(), dynamic.format, tensor_config) {
                Ok(kernel) => kernels.push(Item::Kernel(kernel)),
                Err(message) => {
                    errors.push(super::TranspileError::at_line(message, call.line));
                    out.push(stmt.clone());
                    index += 1;
                    continue;
                }
            }
            match parse_dynamic_replacement_stmts(&kernel_name, &instance, &dynamic, tensor_config) {
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
        match parse_kernel(&kernel_name, &spec, method, tensor_config) {
            Ok(kernel) => kernels.push(Item::Kernel(kernel)),
            Err(message) => {
                errors.push(super::TranspileError::at_line(message, call.line));
                out.push(stmt.clone());
                index += 1;
                continue;
            }
        }
        match parse_replacement_stmts(&kernel_name, &instance, names, &spec, tensor_config) {
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

#[allow(clippy::too_many_arguments)]
fn lower_nested_body(
    body: &[Stmt],
    outer_types: &HashMap<String, Type>,
    kernels: &mut Vec<Item>,
    errors: &mut Vec<super::TranspileError>,
    ordinal: &mut usize,
    used_kernel_names: &mut HashSet<String>,
    used_binding_names: &mut HashSet<String>,
    tensor_config: &TensorLinearConfig,
) -> Vec<Stmt> {
    let mut nested_types = outer_types.clone();
    lower_function_body(
        body,
        &mut nested_types,
        kernels,
        errors,
        ordinal,
        used_kernel_names,
        used_binding_names,
        tensor_config,
    )
}

#[allow(clippy::too_many_arguments)]
fn lower_control_flow_stmt(
    stmt: &Stmt,
    types: &HashMap<String, Type>,
    kernels: &mut Vec<Item>,
    errors: &mut Vec<super::TranspileError>,
    ordinal: &mut usize,
    used_kernel_names: &mut HashSet<String>,
    used_binding_names: &mut HashSet<String>,
    tensor_config: &TensorLinearConfig,
) -> Option<Stmt> {
    let mut lower_body = |body: &[Stmt]| {
        lower_nested_body(body, types, kernels, errors, ordinal, used_kernel_names, used_binding_names, tensor_config)
    };
    Some(match stmt {
        Stmt::If(value) => {
            let mut value = value.clone();
            value.branches = value.branches.into_iter()
                .map(|(condition, body)| (condition, lower_body(&body)))
                .collect();
            value.else_body = value.else_body.map(|body| lower_body(&body));
            Stmt::If(value)
        }
        Stmt::While(value) => {
            let mut value = value.clone();
            value.body = lower_body(&value.body);
            Stmt::While(value)
        }
        Stmt::For(value) => {
            let mut value = value.clone();
            value.body = lower_body(&value.body);
            Stmt::For(value)
        }
        Stmt::Loop(value) => {
            let mut value = value.clone();
            value.body = lower_body(&value.body);
            Stmt::Loop(value)
        }
        Stmt::DoWhile(value) => {
            let mut value = value.clone();
            value.body = lower_body(&value.body);
            Stmt::DoWhile(value)
        }
        Stmt::IfLet(value) => {
            let mut value = value.clone();
            value.then_body = lower_body(&value.then_body);
            value.elif_branches = value.elif_branches.into_iter().map(|mut branch| {
                branch.body = lower_body(&branch.body);
                branch
            }).collect();
            value.else_body = value.else_body.map(|body| lower_body(&body));
            Stmt::IfLet(value)
        }
        Stmt::WhileLet(value) => {
            let mut value = value.clone();
            value.body = lower_body(&value.body);
            Stmt::WhileLet(value)
        }
        Stmt::Match(value) => {
            let mut value = value.clone();
            value.arms = value.arms.into_iter().map(|mut arm| {
                if let MatchBody::Block(body) = arm.body {
                    arm.body = MatchBody::Block(lower_body(&body));
                }
                arm
            }).collect();
            Stmt::Match(value)
        }
        Stmt::Try(value) => {
            let mut value = value.clone();
            value.body = lower_body(&value.body);
            value.catch_clauses = value.catch_clauses.into_iter().map(|mut clause| {
                clause.body = lower_body(&clause.body);
                clause
            }).collect();
            Stmt::Try(value)
        }
        _ => return None,
    })
}

struct DynamicLinearCall<'a> {
    operands: [&'a str; 3],
    bias: Option<&'a str>,
    m: String,
    n: String,
    k: String,
    format: Option<&'a str>,
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
    if method != "linear" { return None; }
    if !matches!(&receiver.kind, ExprKind::Field(gpu, ns)
        if ns == "tensor" && matches!(&gpu.kind, ExprKind::Var(name) if name == "gpu")) { return None; }
    let operand_count = args.iter().position(|arg| arg.label.is_some())?;
    if !matches!(operand_count, 3 | 4) { return None; }
    let raw: Vec<&str> = args[..operand_count].iter().map(|arg| match &arg.value.kind {
        ExprKind::Var(name) if arg.label.is_none() => Some(name.as_str()),
        _ => None,
    }).collect::<Option<_>>()?;
    let dim = |label: &str| args[operand_count..].iter().find(|arg| arg.label.as_deref() == Some(label))
        .and_then(|arg| dimension_source(&arg.value));
    let format = match args[operand_count..].iter().find(|arg| arg.label.as_deref() == Some("format")) {
        None => None,
        Some(arg) => match &arg.value.kind {
            ExprKind::Str(value) if crate::tensor_formats::quantized_linear_geometry(value).is_some() => Some(value.as_str()),
            _ => return None,
        },
    };
    let (operands, bias) = if operand_count == 4 { ([raw[0], raw[1], raw[3]], Some(raw[2])) } else { ([raw[0], raw[1], raw[2]], None) };
    Some(DynamicLinearCall { operands, bias, m: dim("m")?, n: dim("n")?, k: dim("k")?, format })
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

fn parse_kernel(name: &str, spec: &Spec, method: &str, config: &TensorLinearConfig) -> Result<KernelDecl, String> {
    let (tile_rows, tile_cols, _) = fixed_geometry(spec, config);
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

fn parse_dynamic_linear_kernel(name: &str, quals: [GpuQual; 3], has_bias: bool, quantized_format: Option<&str>, tensor_config: &TensorLinearConfig) -> Result<KernelDecl, String> {
    let bias_field = if has_bias { "    let [float32]'global bias\n" } else { "" };
    let bias_param = if has_bias { ", [float32]'global input_bias" } else { "" };
    let bias_assign = if has_bias { "        bias = input_bias\n" } else { "" };
    let initial = if has_bias { "bias[col]" } else { "0.0" };
    let weight_type = if quantized_format.is_some() { "uint8" } else { "float32" };
    let geometry = quantized_format.and_then(crate::tensor_formats::quantized_linear_geometry);
    let block_bytes = geometry.map_or(0, |geometry| geometry.block_bytes);
    let quantized_value = match quantized_format {
        Some("q8_0") => "                let raw = int(b[blockByte + 2 + flat % 32])\n                let quantized = if raw > 127: raw - 256 else: raw\n",
        Some("q5_0") => "                let position = flat % 32\n                let qh = int(b[blockByte + 2]) | (int(b[blockByte + 3]) << 8) | (int(b[blockByte + 4]) << 16) | (int(b[blockByte + 5]) << 24)\n                let packed = int(b[blockByte + 6 + position % 16])\n                let nibble = if position < 16: packed & 0xF else: (packed >> 4) & 0xF\n                let high = (qh >> (position as uint32)) & 1\n                let quantized = (nibble | (high << 4)) - 16\n",
        Some("q4_0") => "                let position = flat % 32\n                let packed = int(b[blockByte + 2 + position % 16])\n                let nibble = if position < 16: packed & 0xF else: (packed >> 4) & 0xF\n                let quantized = nibble - 8\n",
        Some("iq4_nl") => "                let position = flat % 32\n                let packed = int(b[blockByte + 2 + position % 16])\n                let nibble = if position < 16: packed & 0xF else: (packed >> 4) & 0xF\n                let quantized = if nibble == 0: -127 elif nibble == 1: -104 elif nibble == 2: -83 elif nibble == 3: -65 elif nibble == 4: -49 elif nibble == 5: -35 elif nibble == 6: -22 elif nibble == 7: -10 elif nibble == 8: 1 elif nibble == 9: 13 elif nibble == 10: 25 elif nibble == 11: 38 elif nibble == 12: 53 elif nibble == 13: 69 elif nibble == 14: 89 else: 113\n",
        Some("q6_k") => "                let position = flat % 256\n                let iteration = position / 128\n                let within = position % 128\n                let group = within / 32\n                let lane = within % 32\n                let half = lane / 16\n                let qlBase = blockByte + iteration * 64\n                let qhBase = blockByte + 128 + iteration * 32\n                let scaleBase = blockByte + 192 + iteration * 8\n                let low0 = int(b[qlBase + lane])\n                let low32 = int(b[qlBase + lane + 32])\n                let nibble = if group == 0: low0 & 0xF elif group == 1: low32 & 0xF elif group == 2: (low0 >> 4) & 0xF else: (low32 >> 4) & 0xF\n                let highByte = int(b[qhBase + lane])\n                let high = (highByte >> ((group * 2) as uint32)) & 3\n                let scaleRaw = int(b[scaleBase + half + group * 2])\n                let subScale = if scaleRaw > 127: scaleRaw - 256 else: scaleRaw\n                let quantized = subScale * ((nibble | (high << 4)) - 32)\n",
        Some("q3_k") => "                let position = flat % 256\n                let outer = position / 128\n                let remainder = position % 128\n                let group = remainder / 32\n                let within32 = remainder % 32\n                let sub = within32 / 16\n                let lane = within32 % 16\n                let scaleIndex = outer * 8 + group * 2 + sub\n                let scaleWord = scaleIndex / 4\n                let scaleLane = scaleIndex % 4\n                let scaleByte = int(b[blockByte + 96 + (scaleWord % 2) * 4 + scaleLane])\n                let scaleLow = if scaleWord < 2: scaleByte & 0xF else: (scaleByte >> 4) & 0xF\n                let scaleHigh = (int(b[blockByte + 104 + scaleLane]) >> ((scaleWord * 2) as uint32)) & 3\n                let subScale = (scaleLow | (scaleHigh << 4)) - 32\n                let packed = int(b[blockByte + 32 + outer * 32 + sub * 16 + lane])\n                let q = (packed >> ((group * 2) as uint32)) & 3\n                let highMask = int(b[blockByte + sub * 16 + lane]) & ((1 as int) << ((outer * 4 + group) as uint32))\n                let highOffset = if highMask == 0: 4 else: 0\n                let quantized = subScale * (q - highOffset)\n",
        Some("q2_k") => "                let position = flat % 256\n                let outer = position / 128\n                let remainder = position % 128\n                let group = remainder / 32\n                let within32 = remainder % 32\n                let sub = within32 / 16\n                let lane = within32 % 16\n                let scaleByte = int(b[blockByte + outer * 8 + group * 2 + sub])\n                let packed = int(b[blockByte + 16 + outer * 32 + sub * 16 + lane])\n                let q = (packed >> ((group * 2) as uint32)) & 3\n                let minBits = int(b[blockByte + 82]) | (int(b[blockByte + 83]) << 8)\n                let minSign = (minBits >> 15) & 1\n                let minExponent = (minBits >> 10) & 0x1F\n                let minFraction = minBits & 0x3FF\n                var float32 dMin = 0.0\n                if minExponent == 0:\n                    dMin = (minFraction as float32) / 16777216.0\n                else:\n                    dMin = 1.0 + (minFraction as float32) / 1024.0\n                    var int minPower = minExponent - 15\n                    while minPower > 0:\n                        dMin *= 2.0\n                        minPower -= 1\n                    while minPower < 0:\n                        dMin /= 2.0\n                        minPower += 1\n                if minSign == 1:\n                    dMin = 0.0 - dMin\n                minimum = dMin * ((scaleByte >> 4) as float32)\n                let quantized = (scaleByte & 0xF) * q\n",
        _ => "                let position = flat % 256\n                let chunk = position / 64\n                let within = position % 64\n                let half = within / 32\n                let lane = within % 32\n                let subblock = chunk * 2 + half\n                let scalesBase = blockByte + 4\n                let subScale = if subblock < 4: int(b[scalesBase + subblock]) & 63 else: (int(b[scalesBase + subblock + 4]) & 0xF) | ((int(b[scalesBase + subblock - 4]) >> 6) << 4)\n                let subMin = if subblock < 4: int(b[scalesBase + subblock + 4]) & 63 else: (int(b[scalesBase + subblock + 4]) >> 4) | ((int(b[scalesBase + subblock]) >> 6) << 4)\n                let packed = int(b[blockByte + 16 + chunk * 32 + lane])\n                let nibble = if half == 0: packed & 0xF else: (packed >> 4) & 0xF\n                let minBits = int(b[blockByte + 2]) | (int(b[blockByte + 3]) << 8)\n                let minSign = (minBits >> 15) & 1\n                let minExponent = (minBits >> 10) & 0x1F\n                let minFraction = minBits & 0x3FF\n                var float32 dMin = 0.0\n                if minExponent == 0:\n                    dMin = (minFraction as float32) / 16777216.0\n                else:\n                    dMin = 1.0 + (minFraction as float32) / 1024.0\n                    var int minPower = minExponent - 15\n                    while minPower > 0:\n                        dMin *= 2.0\n                        minPower -= 1\n                    while minPower < 0:\n                        dMin /= 2.0\n                        minPower += 1\n                if minSign == 1:\n                    dMin = 0.0 - dMin\n                minimum = dMin * (subMin as float32)\n                let quantized = subScale * nibble\n",
    };
    let block_elements = geometry.map_or(32, |geometry| geometry.block_elements);
    let scale_offset = geometry.map_or(0, |geometry| geometry.scale_offset);
    let product = if quantized_format.is_some() {
        format!("                let flat = col * k + inner\n                let blockByte = (flat / {block_elements}) * {block_bytes}\n                let scaleOffset = blockByte + {scale_offset}\n                let scaleBits = int(b[scaleOffset]) | (int(b[scaleOffset + 1]) << 8)\n                let sign = (scaleBits >> 15) & 1\n                let exponent = (scaleBits >> 10) & 0x1F\n                let fraction = scaleBits & 0x3FF\n                var float32 scale = 0.0\n                if exponent == 0:\n                    scale = (fraction as float32) / 16777216.0\n                elif exponent == 0x1F:\n                    scale = if fraction == 0: 1.0 / 0.0 else: 0.0 / 0.0\n                else:\n                    scale = 1.0 + (fraction as float32) / 1024.0\n                    var int scaleExponent = exponent - 15\n                    while scaleExponent > 0:\n                        scale *= 2.0\n                        scaleExponent -= 1\n                    while scaleExponent < 0:\n                        scale /= 2.0\n                        scaleExponent += 1\n                if sign == 1:\n                    scale = 0.0 - scale\n                var float32 minimum = 0.0\n{quantized_value}                sum += a[row * k + inner] * ((quantized as float32) * scale - minimum)\n")
    } else {
        "                sum += a[row * k + inner] * b[col * k + inner]\n".to_string()
    };
    let warp_product = format!("        {}", product.replace("\n                ", "\n                        "));
    let nested_scalar_product = format!("    {}", product.replace("\n                ", "\n                    "));
    let q8_decode_algorithm = tensor_config.algorithm(Some("q8_0"), true);
    let q5_decode_algorithm = tensor_config.algorithm(Some("q5_0"), true);
    let q4_decode_algorithm = tensor_config.algorithm(Some("q4_0"), true);
    let iq4_decode_algorithm = tensor_config.algorithm(Some("iq4_nl"), true);
    let q8_warp_decode = quantized_format == Some("q8_0") && matches!(q8_decode_algorithm, "auto" | "warp" | "warp-broadcast");
    let q8_scale_lane = if tensor_config.target_warp_width == 32 {
        "0"
    } else {
        "(lane / 32) * 32"
    };
    let q8_scale_inner_guard = if tensor_config.target_warp_width == 32 {
        ""
    } else {
        " and inner < k"
    };
    let q8_value_position = if tensor_config.target_warp_width == 32 {
        "lane"
    } else {
        "flat % 32"
    };
    let q8_value_decode = if tensor_config.target_warp_width == 32 {
        format!("                    let raw = int(b[blockByte + 2 + {q8_value_position}])\n                    let quantized = if raw > 127: raw - 256 else: raw\n                    sum += a[row * k + inner] * ((quantized as float32) * scale)\n")
    } else {
        format!("                    if inner < k:\n                        let raw = int(b[blockByte + 2 + {q8_value_position}])\n                        let quantized = if raw > 127: raw - 256 else: raw\n                        sum += a[row * k + inner] * ((quantized as float32) * scale)\n")
    };
    let q5_value_decode = if tensor_config.target_warp_width == 32 {
        "                    let position = lane\n                    let packed = int(b[blockByte + 6 + position % 16])\n                    let nibble = if position < 16: packed & 0xF else: (packed >> 4) & 0xF\n                    let high = (qh >> (position as uint32)) & 1\n                    let quantized = (nibble | (high << 4)) - 16\n                    sum += a[row * k + inner] * ((quantized as float32) * scale)\n".to_string()
    } else {
        "                    if inner < k:\n                        let position = lane % 32\n                        let packed = int(b[blockByte + 6 + position % 16])\n                        let nibble = if position < 16: packed & 0xF else: (packed >> 4) & 0xF\n                        let high = (qh >> (position as uint32)) & 1\n                        let quantized = (nibble | (high << 4)) - 16\n                        sum += a[row * k + inner] * ((quantized as float32) * scale)\n".to_string()
    };
    let q4_value_decode = if tensor_config.target_warp_width == 32 {
        "                    let position = lane\n                    let packed = int(b[blockByte + 2 + position % 16])\n                    let nibble = if position < 16: packed & 0xF else: (packed >> 4) & 0xF\n                    let quantized = nibble - 8\n                    sum += a[row * k + inner] * ((quantized as float32) * scale)\n".to_string()
    } else {
        "                    if inner < k:\n                        let position = lane % 32\n                        let packed = int(b[blockByte + 2 + position % 16])\n                        let nibble = if position < 16: packed & 0xF else: (packed >> 4) & 0xF\n                        let quantized = nibble - 8\n                        sum += a[row * k + inner] * ((quantized as float32) * scale)\n".to_string()
    };
    let iq4_value_decode = if tensor_config.target_warp_width == 32 {
        "                    let position = lane\n                    let packed = int(b[blockByte + 2 + position % 16])\n                    let nibble = if position < 16: packed & 0xF else: (packed >> 4) & 0xF\n                    let quantized = if nibble == 0: -127 elif nibble == 1: -104 elif nibble == 2: -83 elif nibble == 3: -65 elif nibble == 4: -49 elif nibble == 5: -35 elif nibble == 6: -22 elif nibble == 7: -10 elif nibble == 8: 1 elif nibble == 9: 13 elif nibble == 10: 25 elif nibble == 11: 38 elif nibble == 12: 53 elif nibble == 13: 69 elif nibble == 14: 89 else: 113\n                    sum += a[row * k + inner] * ((quantized as float32) * scale)\n".to_string()
    } else {
        "                    if inner < k:\n                        let position = lane % 32\n                        let packed = int(b[blockByte + 2 + position % 16])\n                        let nibble = if position < 16: packed & 0xF else: (packed >> 4) & 0xF\n                        let quantized = if nibble == 0: -127 elif nibble == 1: -104 elif nibble == 2: -83 elif nibble == 3: -65 elif nibble == 4: -49 elif nibble == 5: -35 elif nibble == 6: -22 elif nibble == 7: -10 elif nibble == 8: 1 elif nibble == 9: 13 elif nibble == 10: 25 elif nibble == 11: 38 elif nibble == 12: 53 elif nibble == 13: 69 elif nibble == 14: 89 else: 113\n                        sum += a[row * k + inner] * ((quantized as float32) * scale)\n".to_string()
    };
    let scale_broadcast = if quantized_format == Some("q8_0") && matches!(q8_decode_algorithm, "auto" | "warp-broadcast") {
        Some((34, &q8_value_decode))
    } else if quantized_format == Some("q4_0") && matches!(q4_decode_algorithm, "auto" | "warp" | "warp-broadcast") {
        Some((18, &q4_value_decode))
    } else if quantized_format == Some("iq4_nl") && matches!(iq4_decode_algorithm, "auto" | "warp" | "warp-broadcast") {
        Some((18, &iq4_value_decode))
    } else {
        None
    };
    let body = if let Some((broadcast_block_bytes, broadcast_value_decode)) = scale_broadcast {
        format!(
            "        if m == 1:\n            let lane = gpu.warp.lane\n            let warpLen = gpu.warp.size\n            let warpInBlock = gpu.thread.x / warpLen\n            let warpsPerBlock = gpu.blockDim.x / warpLen\n            let blockIndex = gpu.block.x + gpu.block.y * gpu.gridDim.x\n            let cell = blockIndex * warpsPerBlock + warpInBlock\n            if cell < n:\n                let row = 0\n                let col = cell\n                var float32 sum = 0.0\n                var int base = 0\n                while base < k:\n                    let inner = base + lane\n                    let scaleLane = {scale_lane}\n                    let flat = col * k + inner\n                    let blockByte = (flat / 32) * {broadcast_block_bytes}\n                    var float32 scale = 0.0\n                    if lane == scaleLane{scale_inner_guard}:\n                        let scaleBits = int(b[blockByte]) | (int(b[blockByte + 1]) << 8)\n                        let sign = (scaleBits >> 15) & 1\n                        let exponent = (scaleBits >> 10) & 0x1F\n                        let fraction = scaleBits & 0x3FF\n                        if exponent == 0:\n                            scale = (fraction as float32) / 16777216.0\n                        elif exponent == 0x1F:\n                            scale = if fraction == 0: 1.0 / 0.0 else: 0.0 / 0.0\n                        else:\n                            scale = 1.0 + (fraction as float32) / 1024.0\n                            var int scaleExponent = exponent - 15\n                            while scaleExponent > 0:\n                                scale *= 2.0\n                                scaleExponent -= 1\n                            while scaleExponent < 0:\n                                scale /= 2.0\n                                scaleExponent += 1\n                        if sign == 1:\n                            scale = 0.0 - scale\n                    scale = gpu.warp.shuffle(scale, scaleLane)\n{value_decode}                    base += warpLen\n                var int offset = warpLen / 2\n                while offset > 0:\n                    sum += gpu.warp.shuffleXor(sum, offset)\n                    offset /= 2\n                if lane == 0:\n                    c[col] = sum + {bias}\n        else:\n            let blockIndex = gpu.block.x + gpu.block.y * gpu.gridDim.x\n            let cell = gpu.thread.x + blockIndex * gpu.blockDim.x\n            if cell < m * n:\n                let row = cell / n\n                let col = cell % n\n                var float32 sum = {initial}\n                for inner in 0..<k:\n{product}                c[row * n + col] = sum\n",
            bias = if has_bias { "bias[col]" } else { "0.0" },
            initial = initial,
            product = nested_scalar_product,
            scale_lane = q8_scale_lane,
            scale_inner_guard = q8_scale_inner_guard,
            value_decode = broadcast_value_decode,
        )
    } else if quantized_format == Some("q5_0") && matches!(q5_decode_algorithm, "auto" | "warp" | "warp-broadcast") {
        format!(
            "        if m == 1:\n            let lane = gpu.warp.lane\n            let warpLen = gpu.warp.size\n            let warpInBlock = gpu.thread.x / warpLen\n            let warpsPerBlock = gpu.blockDim.x / warpLen\n            let blockIndex = gpu.block.x + gpu.block.y * gpu.gridDim.x\n            let cell = blockIndex * warpsPerBlock + warpInBlock\n            if cell < n:\n                let row = 0\n                let col = cell\n                var float32 sum = 0.0\n                var int base = 0\n                while base < k:\n                    let inner = base + lane\n                    let scaleLane = {scale_lane}\n                    let flat = col * k + inner\n                    let blockByte = (flat / 32) * 22\n                    var float32 scale = 0.0\n                    var int qh = 0\n                    if lane == scaleLane{scale_inner_guard}:\n                        let scaleBits = int(b[blockByte]) | (int(b[blockByte + 1]) << 8)\n                        let sign = (scaleBits >> 15) & 1\n                        let exponent = (scaleBits >> 10) & 0x1F\n                        let fraction = scaleBits & 0x3FF\n                        if exponent == 0:\n                            scale = (fraction as float32) / 16777216.0\n                        elif exponent == 0x1F:\n                            scale = if fraction == 0: 1.0 / 0.0 else: 0.0 / 0.0\n                        else:\n                            scale = 1.0 + (fraction as float32) / 1024.0\n                            var int scaleExponent = exponent - 15\n                            while scaleExponent > 0:\n                                scale *= 2.0\n                                scaleExponent -= 1\n                            while scaleExponent < 0:\n                                scale /= 2.0\n                                scaleExponent += 1\n                        if sign == 1:\n                            scale = 0.0 - scale\n                        qh = int(b[blockByte + 2]) | (int(b[blockByte + 3]) << 8) | (int(b[blockByte + 4]) << 16) | (int(b[blockByte + 5]) << 24)\n                    scale = gpu.warp.shuffle(scale, scaleLane)\n                    qh = gpu.warp.shuffle(qh, scaleLane)\n{value_decode}                    base += warpLen\n                var int offset = warpLen / 2\n                while offset > 0:\n                    sum += gpu.warp.shuffleXor(sum, offset)\n                    offset /= 2\n                if lane == 0:\n                    c[col] = sum + {bias}\n        else:\n            let blockIndex = gpu.block.x + gpu.block.y * gpu.gridDim.x\n            let cell = gpu.thread.x + blockIndex * gpu.blockDim.x\n            if cell < m * n:\n                let row = cell / n\n                let col = cell % n\n                var float32 sum = {initial}\n                for inner in 0..<k:\n{product}                c[row * n + col] = sum\n",
            bias = if has_bias { "bias[col]" } else { "0.0" },
            initial = initial,
            product = nested_scalar_product,
            scale_lane = q8_scale_lane,
            scale_inner_guard = q8_scale_inner_guard,
            value_decode = q5_value_decode,
        )
    } else if q8_warp_decode {
        format!(
            "        if m == 1:\n            let lane = gpu.warp.lane\n            let warpLen = gpu.warp.size\n            let warpInBlock = gpu.thread.x / warpLen\n            let warpsPerBlock = gpu.blockDim.x / warpLen\n            let blockIndex = gpu.block.x + gpu.block.y * gpu.gridDim.x\n            let cell = blockIndex * warpsPerBlock + warpInBlock\n            let row = 0\n            let col = cell\n            var float32 sum = 0.0\n            var int inner = lane\n            while inner < k:\n                if cell < n:\n{product}                inner += warpLen\n            var int offset = warpLen / 2\n            while offset > 0:\n                sum += gpu.warp.shuffleXor(sum, offset)\n                offset /= 2\n            if cell < n:\n                if lane == 0:\n                    c[col] = sum + {bias}\n        else:\n            let blockIndex = gpu.block.x + gpu.block.y * gpu.gridDim.x\n            let cell = gpu.thread.x + blockIndex * gpu.blockDim.x\n            if cell < m * n:\n                let row = cell / n\n                let col = cell % n\n                var float32 sum = {initial}\n                for inner in 0..<k:\n{product}                c[row * n + col] = sum\n",
            product = warp_product,
            bias = if has_bias { "bias[col]" } else { "0.0" },
            initial = initial,
        )
    } else {
        format!(
            "        let blockIndex = gpu.block.x + gpu.block.y * gpu.gridDim.x\n        let cell = gpu.thread.x + blockIndex * gpu.blockDim.x\n        if cell < m * n:\n            let row = cell / n\n            let col = cell % n\n            var float32 sum = {initial}\n            for inner in 0..<k:\n{product}            c[row * n + col] = sum\n",
            initial = initial,
            product = product,
        )
    };
    let source = format!(
        "kernel {name}:\n    let [float32]'{qa} a\n    let [{weight_type}]'{qb} b\n{bias_field}    mut [float32]'{qc} c\n    let int m\n    let int n\n    let int k\n    init([float32]'{qa} input_a, [{weight_type}]'{qb} input_b{bias_param}, int input_m, int input_n, int input_k):\n        a = input_a\n        b = input_b\n{bias_assign}        c = [..<input_m * input_n]\n        m = input_m\n        n = input_n\n        k = input_k\n    def ():\n{body}",
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

fn parse_replacement(name: &str, instance: &str, operands: [&str; 3], spec: &Spec, config: &TensorLinearConfig) -> Result<Vec<Item>, String> {
    let (tile_rows, tile_cols, block) = fixed_geometry(spec, config);
    let gx = spec.n.div_ceil(tile_cols);
    let gy = spec.m.div_ceil(tile_rows);
    let source = format!(
        "mut {instance} = {name}({a}, {b}, {c})\nkernel:\n    {instance}(block = {block}, grid = ({gx}, {gy}))\n{c} = {instance}.c\n",
        a = operands[0], b = operands[1], c = operands[2],
    );
    crate::parser::parse(crate::lexer::lex(&source).map_err(|e| format!("tensor dispatch lex error: {e:?}"))?)
        .map(|program| program.items)
        .map_err(|e| format!("tensor dispatch parse error: {}", e.msg()))
}

fn parse_replacement_stmts(name: &str, instance: &str, operands: [&str; 3], spec: &Spec, config: &TensorLinearConfig) -> Result<Vec<Stmt>, String> {
    let (tile_rows, tile_cols, block) = fixed_geometry(spec, config);
    let gx = spec.n.div_ceil(tile_cols);
    let gy = spec.m.div_ceil(tile_rows);
    let source = format!(
        "def __tensor_wrapper():\n    mut {instance} = {name}({a}, {b}, {c})\n    kernel:\n        {instance}(block = {block}, grid = ({gx}, {gy}))\n    {c} = {instance}.c\n",
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
    tensor_config: &TensorLinearConfig,
) -> Result<Vec<Stmt>, String> {
    let bias_guard = call.bias.map(|bias| format!("    guard {bias}.length == {{n}} else throw \"tensor bias length mismatch\"\n")).unwrap_or_default().replace("{n}", &call.n);
    let bias_arg = call.bias.map(|bias| format!("{bias}, ")).unwrap_or_default();
    let weight_guard = if let Some(format) = call.format {
        let geometry = crate::tensor_formats::quantized_linear_geometry(format).unwrap();
        let block_bytes = geometry.block_bytes;
        let block_elements = geometry.block_elements;
        format!("    guard {k} % {block_elements} == 0 else throw \"quantized tensor dimension k must be a multiple of {block_elements}\"\n    guard {b}.length == ({n} * {k} / {block_elements}) * {block_bytes} else throw \"quantized tensor weight length mismatch\"\n", k = call.k, b = call.operands[1], n = call.n)
    } else {
        format!("    guard {b}.length == {n} * {k} else throw \"tensor weight length mismatch\"\n", b = call.operands[1], n = call.n, k = call.k)
    };
    let decode_algorithm = tensor_config.algorithm(call.format, true);
    let warp_decode = match call.format {
        Some("q8_0" | "q5_0" | "q4_0") => matches!(decode_algorithm, "auto" | "warp" | "warp-broadcast"),
        Some("iq4_nl") => matches!(decode_algorithm, "auto" | "warp" | "warp-broadcast"),
        _ => false,
    };
    let warps_per_block = 256 / tensor_config.target_warp_width.max(1);
    let dispatch = if warp_decode {
        format!("    let {instance}_blocks = if {m} == 1: ({n} + {tail}) / {warps_per_block} else: ({m} * {n} + 255) / 256\n", instance = instance, m = call.m, n = call.n, tail = warps_per_block - 1)
    } else {
        String::new()
    };
    let grid = if warp_decode {
        format!("{instance}_blocks")
    } else {
        format!("({m} * {n} + 255) / 256", m = call.m, n = call.n)
    };
    let source = format!(
        "def __tensor_wrapper() throws:\n    guard {m} > 0 else throw \"tensor dimension m must be positive\"\n    guard {n} > 0 else throw \"tensor dimension n must be positive\"\n    guard {k} > 0 else throw \"tensor dimension k must be positive\"\n    guard {a}.length == {m} * {k} else throw \"tensor left operand length mismatch\"\n{weight_guard}{bias_guard}    guard {c}.length == {m} * {n} else throw \"tensor destination length mismatch\"\n    mut {instance} = {name}({a}, {b}, {bias_arg}{m}, {n}, {k})\n{dispatch}    kernel:\n        {instance}(block = 256, grid = {grid})\n    {c} = {instance}.c\n",
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
        Stmt::IfLet(value) => value.then_body.iter().any(stmt_uses_host_tensor)
            || value.elif_branches.iter().any(|branch| branch.body.iter().any(stmt_uses_host_tensor))
            || value.else_body.as_ref().is_some_and(|body| body.iter().any(stmt_uses_host_tensor)),
        Stmt::WhileLet(value) => value.body.iter().any(stmt_uses_host_tensor),
        Stmt::Match(value) => value.arms.iter().any(|arm| matches!(&arm.body, MatchBody::Block(body) if body.iter().any(stmt_uses_host_tensor))),
        Stmt::Try(value) => value.body.iter().any(stmt_uses_host_tensor)
            || value.catch_clauses.iter().any(|clause| clause.body.iter().any(stmt_uses_host_tensor)),
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
    fn lowers_host_scheduling_inside_control_flow() {
        let lowered = lower(&parse("def work() throws:\n    let [float32, k = 2, m = 2]'gpu'global a = [0.0 for ..<4]\n    let [float32, n = 2, k = 2]'gpu'global b = [0.0 for ..<4]\n    mut [float32, n = 2, m = 2]'gpu'unified c = [0.0 for ..<4]\n    if true:\n        gpu.tensor.matmul(a, b, c)\n"));
        assert!(lowered.errors.is_empty(), "{:?}", lowered.errors);
        let function = lowered.program.items.iter().find_map(|item| match item {
            Item::Fn(function) if function.name == "work" => Some(function),
            _ => None,
        }).unwrap();
        let Stmt::If(branch) = &function.body[3] else { panic!("expected if") };
        assert!(branch.branches[0].1.iter().any(|stmt| matches!(stmt, Stmt::KernelBlock(_))));
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

    #[test]
    fn native_fixed_geometry_requires_aligned_dimensions() {
        let mut config = TensorLinearConfig::default();
        config.native_fixed_matrices = true;
        let spec = Spec {
            m: 64,
            n: 32,
            k: 16,
            quals: [GpuQual::Global, GpuQual::Global, GpuQual::Unified],
            transpose_b: false,
        };
        assert_eq!(fixed_geometry(&spec, &config), (8, 8, 32));
        let unaligned = Spec { k: 15, ..spec.clone() };
        assert_eq!(fixed_geometry(&unaligned, &config), (16, 16, 256));
        config.native_fixed_matrices = false;
        assert_eq!(fixed_geometry(&spec, &config), (16, 16, 256));
    }
}
