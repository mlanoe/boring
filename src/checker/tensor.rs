//! Target-independent resolution for the proposed block-collective tensor API.
//! Direct kernel call statements are validated before interpretation or code
//! generation. The scalar backend and interpreter fallbacks consume the same
//! resolved operation.
use crate::ast::*;
use crate::errors::SourceError;

pub(crate) fn is_tensor_call(call: &Expr) -> bool {
    matches!(&call.kind, ExprKind::MethodCall(receiver, _, _)
        if matches!(&receiver.kind, ExprKind::Field(gpu, name)
            if name == "tensor" && matches!(&gpu.kind, ExprKind::Var(n) if n == "gpu")))
}

impl super::Checker {
    pub(super) fn check_direct_tensor_calls(&mut self, kernel: &KernelDecl) {
        self.errors.extend(validate_kernel(kernel));
    }

    pub(super) fn check_host_tensor_call(&mut self, call: &Expr) {
        let ExprKind::MethodCall(_, method, args) = &call.kind else { return };
        if !matches!(method.as_str(), "matmul" | "mma" | "linear") {
            self.error(
                "host gpu.tensor calls must use matmul, mma, or linear; tile operations require a kernel entry point",
                call.line,
                call.col,
            );
            return;
        }
        let operand_count = args.iter().position(|arg| arg.label.is_some()).unwrap_or(args.len());
        let labeled = &args[operand_count..];
        let dynamic_linear = method == "linear"
            && matches!(operand_count, 3 | 4)
            && args[..operand_count].iter().all(|arg| arg.label.is_none())
            && matches!(labeled.len(), 3 | 4)
            && labeled.iter().all(|arg| arg.label.as_deref().is_some_and(|label| matches!(label, "m" | "n" | "k" | "format")));
        if dynamic_linear {
            let mut labels = std::collections::HashSet::new();
            if labeled.iter().any(|arg| !labels.insert(arg.label.as_deref().unwrap()))
                || ["m", "n", "k"].iter().any(|label| !labels.contains(label)) {
                self.error("dynamic gpu.tensor.linear requires distinct m, n, and k arguments (and at most one format)", call.line, call.col);
                return;
            }
            let format = labeled.iter().find(|arg| arg.label.as_deref() == Some("format"));
            let quantized = match format.map(|arg| &arg.value.kind) {
                None => false,
                Some(ExprKind::Str(value)) if crate::tensor_formats::quantized_linear_geometry(value).is_some() => true,
                Some(_) => {
                    self.error(
                        format!("unsupported quantized tensor linear format; expected one of: {}", crate::tensor_formats::SUPPORTED_QUANTIZED_LINEAR_FORMATS),
                        call.line,
                        call.col,
                    );
                    return;
                }
            };
            for (index, arg) in args[..operand_count].iter().enumerate() {
                let ExprKind::Var(name) = &arg.value.kind else {
                    self.error("host tensor operands must be direct variables", call.line, call.col);
                    return;
                };
                let Some(binding) = self.lookup(name) else {
                    self.error(format!("undefined tensor operand '{name}'"), call.line, call.col);
                    return;
                };
                let Some(ty) = binding.ty.as_ref() else {
                    self.error("dynamic tensor operands require explicit array types", call.line, call.col);
                    return;
                };
                let mut ty = ty.without_mut();
                let Type::Qualified(inner, qualifier) = ty else {
                    self.error("dynamic tensor operands require gpu global or unified storage", call.line, call.col);
                    return;
                };
                if !matches!(qualifier, OwnerQual::GpuGlobal | OwnerQual::GpuUnified) {
                    self.error("dynamic tensor operands require gpu global or unified storage", call.line, call.col);
                    return;
                }
                ty = inner.without_mut();
                let elem = match ty {
                    Type::Array(elem) | Type::LabeledArray(elem, _) => elem.without_mut(),
                    _ => {
                        self.error("dynamic tensor operands require float32 arrays", call.line, call.col);
                        return;
                    }
                };
                let float32 = matches!(elem, Type::Float32)
                    || matches!(elem, Type::Named(name) if name == "float32" || name == "Float32" || name == "f32");
                let uint8 = matches!(elem, Type::Uint8)
                    || matches!(elem, Type::Named(name) if name == "uint8" || name == "Uint8" || name == "u8");
                let expected = if quantized && index == 1 { uint8 } else { float32 };
                if !expected {
                    self.error(if quantized && index == 1 {
                        "quantized tensor weights require a uint8 array"
                    } else {
                        "dynamic tensor operands require float32 arrays"
                    }, call.line, call.col);
                    return;
                }
            }
            let output_index = operand_count - 1;
            let ExprKind::Var(output) = &args[output_index].value.kind else { unreachable!() };
            if !self.lookup(output).is_some_and(|binding| binding.kind.is_mutable()) {
                self.error("tensor destination must be mutable", call.line, call.col);
            }
            if args[..output_index].iter().any(|arg| matches!(&arg.value.kind, ExprKind::Var(name) if name == output)) {
                self.error("tensor destination must not alias an input", call.line, call.col);
            }
            return;
        }
        if args.len() != 3 || args.iter().any(|arg| arg.label.is_some() || arg.spread || arg.default_rest) {
            self.error(
                "gpu.tensor.matmul and mma require three positional operands; linear accepts three positional operands, or three/four operands followed by m, n, and k",
                call.line,
                call.col,
            );
            return;
        }
        let mut operands = Vec::with_capacity(3);
        for arg in args {
            let ExprKind::Var(name) = &arg.value.kind else {
                self.error("host tensor operands must be direct variables", call.line, call.col);
                return;
            };
            let Some(binding) = self.lookup(name) else {
                self.error(format!("undefined tensor operand '{name}'"), call.line, call.col);
                return;
            };
            let Some(ty) = binding.ty.clone() else {
                self.error("host tensor operands require explicit labeled-array types", call.line, call.col);
                return;
            };
            operands.push((name.clone(), ty, binding.kind.is_mutable()));
        }
        let refs = std::array::from_fn(|i| {
            let (name, ty, mutable) = &operands[i];
            (name.as_str(), ty, *mutable)
        });
        if let Err(error) = resolve_host_types(method, refs, call) {
            self.errors.push(error);
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HostOperation {
    pub accumulate: bool,
    pub transpose_b: bool,
    pub operands: [String; 3],
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

pub(crate) fn resolve_host_types(
    method: &str,
    operands: [(&str, &Type, bool); 3],
    call: &Expr,
) -> Result<HostOperation, SourceError> {
    let error = |message: &str| SourceError::new(message, call.line, call.col, call.len);
    let (accumulate, transpose_b) = match method {
        "matmul" => (false, false),
        "mma" => (true, false),
        "linear" => (false, true),
        _ => return Err(error("expected gpu.tensor.matmul, gpu.tensor.mma, or gpu.tensor.linear")),
    };
    if !operands[2].2 {
        return Err(error("tensor destination must be mutable"));
    }
    if operands[2].0 == operands[0].0 || operands[2].0 == operands[1].0 {
        return Err(error("tensor destination must not alias an input"));
    }
    let mut shapes = Vec::with_capacity(3);
    for (_, ty, _) in &operands {
        let Type::Qualified(inner, qualifier) = ty.without_mut() else {
            return Err(error("host tensor operands require gpu global or unified storage"));
        };
        if !matches!(qualifier, OwnerQual::GpuGlobal | OwnerQual::GpuUnified) {
            return Err(error("host tensor operands require gpu global or unified storage"));
        }
        let Type::LabeledArray(elem, axes) = inner.without_mut() else {
            return Err(error("host tensor operands require labeled rank-two arrays"));
        };
        let float32 = matches!(elem.without_mut(), Type::Float32)
            || matches!(elem.without_mut(), Type::Named(name) if name == "float32" || name == "Float32" || name == "f32");
        if !float32 || axes.len() != 2 {
            return Err(error("host tensor operands require rank-two float32 arrays"));
        }
        let extents: Option<Vec<usize>> = axes
            .iter()
            .map(|axis| axis.size.as_ref().and_then(|size| positive_codegen_literal(&size.0)))
            .collect();
        let extents = extents.ok_or_else(|| {
            error("initial host tensor shapes require positive i32-sized literal extents")
        })?;
        extents[0]
            .checked_mul(extents[1])
            .filter(|size| *size <= i32::MAX as usize)
            .ok_or_else(|| error("tensor shape overflows addressable size"))?;
        shapes.push(extents);
    }
    let (k, m) = (shapes[0][0], shapes[0][1]);
    let n = if transpose_b { shapes[1][1] } else { shapes[1][0] };
    let right_k = if transpose_b { shapes[1][0] } else { shapes[1][1] };
    if right_k != k || shapes[2] != [n, m] {
        return Err(error(
            if transpose_b {
                "incompatible tensor shapes: expected A[K,M], W[K,N], C[N,M]"
            } else {
                "incompatible tensor shapes: expected A[K,M], B[N,K], C[N,M]"
            },
        ));
    }
    Ok(HostOperation {
        accumulate,
        transpose_b,
        operands: std::array::from_fn(|i| operands[i].0.to_string()),
        m,
        n,
        k,
    })
}

fn validate_kernel(kernel: &KernelDecl) -> Vec<SourceError> {
    let mut errors = Vec::new();
    for init in &kernel.inits {
        if stmts_use_tensor(&init.body) {
            errors.push(SourceError::at(
                "tensor tile calls are not allowed in constructors",
                kernel.line,
                kernel.col,
            ));
        }
    }
    for field in &kernel.fields {
        if field.default.as_ref().is_some_and(expr_uses_tensor) {
            errors.push(SourceError::at(
                "tensor tile calls are not allowed in field initializers",
                field.line,
                field.col,
            ));
        }
    }
    for method in &kernel.methods {
        if !stmts_use_tensor(&method.body) {
            continue;
        }
        if !method.name.is_empty() {
            errors.push(SourceError::at(
                "tensor tile calls must be in the kernel entry point, not a helper method",
                method.line,
                method.col,
            ));
            continue;
        }
        let mut uniform = std::collections::HashSet::new();
        let mut shadowed = std::collections::HashSet::new();
        let mut safe_prefix = true;
        for stmt in &method.body {
            if stmt_uses_tensor(stmt) {
                let Stmt::Expr(call) = stmt else {
                    errors.push(SourceError::at("tensor calls must be standalone entry-point statements, outside branches and loops", method.line, method.col));
                    safe_prefix = false;
                    continue;
                };
                if !is_tensor_call(call) {
                    errors.push(SourceError::new(
                        "tensor calls cannot be nested in another expression",
                        call.line,
                        call.col,
                        call.len,
                    ));
                    safe_prefix = false;
                    continue;
                }
                let checked = resolve(call, kernel).and_then(|op| {
                    let error = |msg| SourceError::new(msg, call.line, call.col, call.len);
                    if shadowed.contains("gpu") || op.operands.iter().any(|n| shadowed.contains(n)) {
                        return Err(error("tensor operands and gpu namespace must not be shadowed"));
                    }
                    if !safe_prefix {
                        return Err(error("tensor participation cannot be proven after control flow or side effects"));
                    }
                    if !uniform_coordinate(&op.row, &uniform) || !uniform_coordinate(&op.col, &uniform) {
                        return Err(error("tensor origins must be uniform non-negative integer expressions"));
                    }
                    Ok(op)
                });
                if let Err(error) = checked {
                    errors.push(error);
                }
            } else {
                match stmt {
                    Stmt::Comment(_) => {}
                    Stmt::Let(binding) => {
                        let valid = matches!(binding.binding, BindingKind::Let)
                            && !binding.is_lazy
                            && binding.ty.as_ref().is_none_or(|ty| {
                                matches!(ty, Type::Int)
                                    || matches!(ty, Type::Named(n) if n == "int")
                            })
                            && binding
                                .value
                                .as_ref()
                                .is_some_and(|e| uniform_coordinate(e, &uniform));
                        uniform.remove(&binding.name);
                        shadowed.insert(binding.name.clone());
                        if valid {
                            uniform.insert(binding.name.clone());
                        } else {
                            safe_prefix = false;
                        }
                    }
                    _ => safe_prefix = false,
                }
            }
        }
    }
    errors
}

// Deliberately conservative: no lane values, memory reads, calls, casts,
// subtraction or division. A richer range/uniformity analysis can extend this.
fn uniform_coordinate(expr: &Expr, locals: &std::collections::HashSet<String>) -> bool {
    match &expr.kind {
        ExprKind::Int(value) => *value >= 0,
        ExprKind::Var(name) => locals.contains(name),
        ExprKind::BinOp(BinOp::Add | BinOp::Mul, a, b) => {
            uniform_coordinate(a, locals) && uniform_coordinate(b, locals)
        }
        ExprKind::Field(base, axis) if matches!(axis.as_str(), "x" | "y" | "z") => {
            matches!(&base.kind, ExprKind::Field(gpu, member)
                if matches!(member.as_str(), "block" | "blockDim" | "gridDim" | "block_dim" | "grid_dim")
                && matches!(&gpu.kind, ExprKind::Var(name) if name == "gpu"))
        }
        _ => false,
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct TileOperation {
    pub accumulate: bool,
    pub transpose_b: bool,
    pub operands: [String; 3],
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub row: Expr,
    pub col: Expr,
    pub rows: usize,
    pub cols: usize,
}

/// Resolve direct field operands without inferring shape from label spellings.
/// Coordinates remain expressions: uniformity is a separate collective check.
pub(crate) fn resolve(call: &Expr, kernel: &KernelDecl) -> Result<TileOperation, SourceError> {
    resolve_fields(call, &kernel.fields)
}

pub(crate) fn resolve_fields(
    call: &Expr,
    fields: &[KernelFieldDecl],
) -> Result<TileOperation, SourceError> {
    let error = |message: &str| SourceError::new(message, call.line, call.col, call.len);
    let ExprKind::MethodCall(receiver, method, args) = &call.kind else {
        return Err(error("expected a gpu.tensor tile operation"));
    };
    if !matches!(&receiver.kind, ExprKind::Field(gpu, name)
        if name == "tensor" && matches!(&gpu.kind, ExprKind::Var(n) if n == "gpu"))
    {
        return Err(error("expected the explicit gpu.tensor namespace"));
    }
    let (accumulate, transpose_b) = match method.as_str() {
        "matmulTile" => (false, false),
        "mmaTile" => (true, false),
        "linearTile" => (false, true),
        _ => return Err(error("expected matmulTile, mmaTile, or linearTile")),
    };
    if args.len() != 7 || args.iter().any(|a| a.spread || a.default_rest) {
        return Err(error(
            "tensor tile calls require three operands and row, col, rows, cols",
        ));
    }
    let available_fields = fields;
    let mut fields = Vec::new();
    for arg in &args[..3] {
        if arg.label.is_some() {
            return Err(error("tensor operands must be positional kernel fields"));
        }
        let ExprKind::Var(name) = &arg.value.kind else {
            return Err(error("tensor operands must be direct kernel fields"));
        };
        let field = available_fields
            .iter()
            .find(|f| f.name == *name)
            .ok_or_else(|| error("tensor operand is not a kernel field"))?;
        if !matches!(field.qual, GpuQual::Global | GpuQual::Unified) {
            return Err(error("tensor operands require global or unified storage"));
        }
        fields.push(field);
    }
    if !matches!(fields[2].binding, FieldBinding::Mut | FieldBinding::Var) {
        return Err(error("tensor destination must be mutable"));
    }
    if fields[2].name == fields[0].name || fields[2].name == fields[1].name {
        return Err(error("tensor destination must not alias an input"));
    }
    let mut shapes = Vec::new();
    for field in &fields {
        let Type::LabeledArray(elem, axes) = field.ty.without_mut() else {
            return Err(error("tensor operands require labeled rank-two arrays"));
        };
        let float32 = matches!(elem.without_mut(), Type::Float32)
            || matches!(elem.without_mut(), Type::Named(name) if name == "float32");
        if !float32 || axes.len() != 2 {
            return Err(error("tensor operands require rank-two float32 arrays"));
        }
        let extents: Option<Vec<usize>> = axes
            .iter()
            .map(|axis| {
                axis.size
                    .as_ref()
                    .and_then(|size| positive_codegen_literal(&size.0))
            })
            .collect();
        let extents = extents.ok_or_else(|| {
            error("initial tensor shapes require positive i32-sized literal extents")
        })?;
        extents[0]
            .checked_mul(extents[1])
            .filter(|size| *size <= i32::MAX as usize)
            .ok_or_else(|| error("tensor shape overflows addressable size"))?;
        shapes.push(extents);
    }
    let (k, m) = (shapes[0][0], shapes[0][1]);
    let n = if transpose_b { shapes[1][1] } else { shapes[1][0] };
    let right_k = if transpose_b { shapes[1][0] } else { shapes[1][1] };
    if right_k != k || shapes[2] != [n, m] {
        return Err(error(
            if transpose_b {
                "incompatible tensor shapes: expected A[K,M], W[K,N], C[N,M]"
            } else {
                "incompatible tensor shapes: expected A[K,M], B[N,K], C[N,M]"
            },
        ));
    }
    let mut named = std::collections::HashMap::new();
    for arg in &args[3..] {
        let Some(label) = arg.label.as_deref() else {
            return Err(error("tile coordinates and extents must be named"));
        };
        if !matches!(label, "row" | "col" | "rows" | "cols")
            || named.insert(label, &arg.value).is_some()
        {
            return Err(error("unknown or duplicate tensor tile argument"));
        }
    }
    let rows = positive_codegen_literal(named["rows"])
        .ok_or_else(|| error("rows must be a positive i32-sized integer literal"))?;
    let cols = positive_codegen_literal(named["cols"])
        .ok_or_else(|| error("cols must be a positive i32-sized integer literal"))?;
    rows.checked_mul(cols)
        .filter(|size| *size <= i32::MAX as usize)
        .ok_or_else(|| error("tensor tile size overflows addressable size"))?;
    for key in ["row", "col"] {
        if matches!(named[key].kind, ExprKind::Int(v) if v < 0) {
            return Err(error("tensor tile coordinates must be non-negative"));
        }
    }
    Ok(TileOperation {
        accumulate,
        transpose_b,
        operands: std::array::from_fn(|i| fields[i].name.clone()),
        m,
        n,
        k,
        row: named["row"].clone(),
        col: named["col"].clone(),
        rows,
        cols,
    })
}

// WGSL is the narrowest common backend: the scalar fallback deliberately uses
// signed i32 arithmetic before converting a proven non-negative index to u32.
fn positive_codegen_literal(expr: &Expr) -> Option<usize> {
    if let ExprKind::Int(value) = expr.kind {
        usize::try_from(value)
            .ok()
            .filter(|v| *v > 0 && *v <= i32::MAX as usize)
    } else {
        None
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    fn resolve_source(changes: &[(&str, &str)]) -> Result<TileOperation, SourceError> {
        let mut src = "kernel Matrix:\n    let [float32, inner = 5, height = 3]'global a\n    let [float32, width = 7, inner = 5]'unified b\n    mut [float32, width = 7, height = 3]'global c\n    def ():\n        gpu.tensor.matmulTile(a, b, c, row = 0, col = 0, rows = 4, cols = 8)\n".to_string();
        for (from, to) in changes {
            src = src.replace(from, to);
        }
        let program = crate::parser::parse(crate::lexer::lex(&src).unwrap()).unwrap();
        let Item::Kernel(kernel) = &program.items[0] else {
            panic!("kernel");
        };
        let Stmt::Expr(call) = &kernel.methods[0].body[0] else {
            panic!("call");
        };
        resolve(call, kernel)
    }
    #[test]
    fn tensor_rectangular_tail_and_axis_names() {
        let op = resolve_source(&[]).unwrap();
        assert_eq!((op.m, op.n, op.k, op.rows, op.cols), (3, 7, 5, 4, 8));
        assert!(!op.accumulate);
        assert_eq!(op.operands, ["a", "b", "c"]);
        assert!(
            resolve_source(&[("matmulTile", "mmaTile")])
                .unwrap()
                .accumulate
        );
        let linear = resolve_source(&[
            ("[float32, width = 7, inner = 5]'unified b", "[float32, inner = 5, width = 7]'unified b"),
            ("matmulTile", "linearTile"),
        ]).unwrap();
        assert!(linear.transpose_b);
        assert_eq!((linear.m, linear.n, linear.k), (3, 7, 5));
    }
    #[test]
    fn tensor_rejects_host_helpers_and_constructors() {
        let call = "gpu.tensor.matmulTile(a, b, c, row = 0, col = 0, rows = 4, cols = 8)";
        for (source, expected) in [
            (format!("{call}\n"), "tile operations require a kernel"),
            (format!("kernel K:\n    def helper():\n        {call}\n    def ():\n        let x = 0\n"), "not a helper method"),
            (format!("kernel K:\n    init():\n        {call}\n    def ():\n        let x = 0\n"), "not allowed in constructors"),
        ] {
            let program = crate::parser::parse(crate::lexer::lex(&source).unwrap()).unwrap();
            for result in [super::super::check(&program), super::super::check_kernel_dispatch_only(&program)] {
                assert!(result.errors.iter().any(|e| e.message.contains(expected)), "expected {expected}: {:?}", result.errors);
            }
        }
    }

    #[test]
    fn tensor_rejects_invalid_contracts() {
        for (changes, expected) in [
            (vec![("mut [float32", "let [float32")], "mutable"),
            (vec![("a, b, c,", "a, b, a,")], "mutable"),
            (
                vec![("inner = 5]'unified", "inner = 6]'unified")],
                "incompatible",
            ),
            (vec![("float32", "float64")], "float32"),
            (vec![("'global a", "'actor a")], "storage"),
            (vec![("cols = 8", "rows = 8")], "duplicate"),
            (vec![("rows = 4", "rows = 0")], "positive"),
            (vec![("rows = 4", "rows = 2147483648")], "i32-sized"),
            (vec![(", cols = 8", "")], "require three"),
            (vec![("a, b, c,", "a, b, missing,")], "not a kernel field"),
        ] {
            let err = resolve_source(&changes).unwrap_err();
            assert!(err.message.contains(expected), "{}", err.message);
        }
    }

    #[test]
    fn host_tensor_contract_is_checked() {
        let source = "let [float32, k = 5, m = 3]'gpu'global a = [float32(i) for i in 0..<15]\nlet [float32, n = 7, k = 5]'gpu'unified b = [float32(i) for i in 0..<35]\nmut [float32, n = 7, m = 3]'gpu'unified c = [0.0 for ..<21]\ngpu.tensor.matmul(a, b, c)\n";
        let program = crate::parser::parse(crate::lexer::lex(source).unwrap()).unwrap();
        let result = super::super::check(&program);
        assert!(result.errors.is_empty(), "{:?}", result.errors);

        let immutable = source.replace("mut [float32, n", "let [float32, n");
        let program = crate::parser::parse(crate::lexer::lex(&immutable).unwrap()).unwrap();
        let result = super::super::check(&program);
        assert!(result.errors.iter().any(|e| e.message.contains("destination must be mutable")));

        let build_result = super::super::check_kernel_dispatch_only(
            &crate::parser::parse(crate::lexer::lex(source).unwrap()).unwrap(),
        );
        assert!(build_result.errors.is_empty(), "{:?}", build_result.errors);

        let linear = source
            .replace("[float32, n = 7, k = 5]'gpu'unified b", "[float32, k = 5, n = 7]'gpu'unified b")
            .replace("gpu.tensor.matmul", "gpu.tensor.linear");
        let result = super::super::check(
            &crate::parser::parse(crate::lexer::lex(&linear).unwrap()).unwrap(),
        );
        assert!(result.errors.is_empty(), "{:?}", result.errors);
    }

    #[test]
    fn dynamic_linear_accepts_flat_gpu_arrays_and_named_dimensions() {
        let source = "req [float32]'gpu'unified compute([float32]'gpu'global x, [float32]'gpu'global w, int m, int n, int k) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32 for ..<m * n]\n    gpu.tensor.linear(x, w, y, m = m, n = n, k = k)\n    y\n";
        let program = crate::parser::parse(crate::lexer::lex(source).unwrap()).unwrap();
        let result = super::super::check(&program);
        assert!(result.errors.is_empty(), "{:?}", result.errors);

        let duplicate = source.replace("n = n", "m = n");
        let result = super::super::check(
            &crate::parser::parse(crate::lexer::lex(&duplicate).unwrap()).unwrap(),
        );
        assert!(result.errors.iter().any(|error| error.message.contains("distinct m, n, and k")));

        let immutable = source.replace("mut [float32]'gpu'unified y", "let [float32]'gpu'unified y");
        let result = super::super::check(
            &crate::parser::parse(crate::lexer::lex(&immutable).unwrap()).unwrap(),
        );
        assert!(result.errors.iter().any(|error| error.message.contains("destination must be mutable")));

        let biased = source
            .replace("int m", "[float32]'gpu'global bias, int m")
            .replace("linear(x, w, y,", "linear(x, w, bias, y,");
        let result = super::super::check(
            &crate::parser::parse(crate::lexer::lex(&biased).unwrap()).unwrap(),
        );
        assert!(result.errors.is_empty(), "{:?}", result.errors);
    }
}

fn expr_uses_tensor(e: &Expr) -> bool {
    if matches!(&e.kind, ExprKind::Field(gpu, name) if name == "tensor" && matches!(&gpu.kind, ExprKind::Var(n) if n == "gpu"))
    {
        return true;
    }
    match &e.kind {
        ExprKind::Int(_)
        | ExprKind::UInt64(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Bool(_)
        | ExprKind::Nil
        | ExprKind::Void
        | ExprKind::Var(_)
        | ExprKind::DotIdent(_) => false,
        ExprKind::StringInterp(segs) => segs.iter().any(|s| match s {
            StringSegment::Lit(_) => false,
            StringSegment::Expr(e) => expr_uses_tensor(e),
            StringSegment::FormattedExpr(e, _) => expr_uses_tensor(e),
        }),
        ExprKind::BinOp(_, l, r) => expr_uses_tensor(l) || expr_uses_tensor(r),
        ExprKind::UnaryOp(_, x) => expr_uses_tensor(x),
        ExprKind::Assign(l, r) | ExprKind::QuestionAssign(l, r) => {
            expr_uses_tensor(l) || expr_uses_tensor(r)
        }
        ExprKind::Field(obj, _) | ExprKind::OptionalField(obj, _) => expr_uses_tensor(obj),
        ExprKind::Index(a, i) => expr_uses_tensor(a) || expr_uses_tensor(i),
        ExprKind::LabeledIndex(a, args) => {
            expr_uses_tensor(a) || args.iter().any(|arg| expr_uses_tensor(&arg.value))
        }
        ExprKind::Call(callee, args) => {
            expr_uses_tensor(callee) || args.iter().any(|a| expr_uses_tensor(&a.value))
        }
        ExprKind::MethodCall(obj, _, args) | ExprKind::OptionalMethodCall(obj, _, args) => {
            expr_uses_tensor(obj) || args.iter().any(|a| expr_uses_tensor(&a.value))
        }
        ExprKind::GenericCall(callee, _, args) => {
            expr_uses_tensor(callee) || args.iter().any(|a| expr_uses_tensor(&a.value))
        }
        ExprKind::Pipe(lhs, _, args) => {
            expr_uses_tensor(lhs) || args.iter().any(|a| expr_uses_tensor(&a.value))
        }
        ExprKind::New { arena, ctor } => {
            arena.as_ref().map(|a| expr_uses_tensor(a)).unwrap_or(false) || expr_uses_tensor(ctor)
        }
        ExprKind::KernelLaunch { config, kernel } => {
            expr_uses_tensor(kernel)
                || config.block.as_ref().map(expr_uses_tensor).unwrap_or(false)
                || config.grid.as_ref().map(expr_uses_tensor).unwrap_or(false)
                || config.after.as_ref().map(expr_uses_tensor).unwrap_or(false)
        }
        ExprKind::TryElse(a, b) => expr_uses_tensor(a) || expr_uses_tensor(b),
        ExprKind::TryElseBlock(body, else_body) => {
            stmts_use_tensor(body) || stmts_use_tensor(else_body)
        }
        ExprKind::Array(items) | ExprKind::Tuple(items) | ExprKind::Set(items) => {
            items.iter().any(expr_uses_tensor)
        }
        ExprKind::ArrayFill { value, count } => expr_uses_tensor(value) || expr_uses_tensor(count),
        ExprKind::ArrayAlloc { count } => expr_uses_tensor(count),
        ExprKind::ArrayComp { expr, count, .. } => {
            expr_uses_tensor(expr) || expr_uses_tensor(count)
        }
        ExprKind::ArrayCompIter { expr, iter, .. } => {
            expr_uses_tensor(expr) || expr_uses_tensor(iter)
        }
        ExprKind::LabeledArrayComp { expr, clauses } => {
            expr_uses_tensor(expr) || clauses.iter().any(|(_, count)| expr_uses_tensor(count))
        }
        ExprKind::RelabelCast(x, _) => expr_uses_tensor(x),
        ExprKind::TrailingArrayBlock {
            callee, args, body, ..
        } => {
            expr_uses_tensor(callee)
                || args.iter().any(|a| expr_uses_tensor(&a.value))
                || stmts_use_tensor(body)
        }
        ExprKind::Dict(pairs) => pairs
            .iter()
            .any(|(k, v)| expr_uses_tensor(k) || expr_uses_tensor(v)),
        ExprKind::Range { start, end, .. } => expr_uses_tensor(start) || expr_uses_tensor(end),
        ExprKind::SliceRange { start, end, .. } => {
            start.as_ref().map(|s| expr_uses_tensor(s)).unwrap_or(false)
                || end.as_ref().map(|e| expr_uses_tensor(e)).unwrap_or(false)
        }
        ExprKind::Cast(x, _) => expr_uses_tensor(x),
        ExprKind::Else(a, b) => expr_uses_tensor(a) || expr_uses_tensor(b),
        ExprKind::Closure(_, _, body, _, _) => match body {
            ClosureBody::Expr(e) => expr_uses_tensor(e),
            ClosureBody::Block(stmts) => stmts_use_tensor(stmts),
        },
        ExprKind::If(s) => if_stmt_uses_tensor(s),
        ExprKind::Match(s) => match_stmt_uses_tensor(s),
        ExprKind::Block(stmts) | ExprKind::Do(stmts) => stmts_use_tensor(stmts),
        ExprKind::Loop(s) => stmts_use_tensor(&s.body),
        ExprKind::Task(x) => expr_uses_tensor(x),
        ExprKind::TaskWithTimeout(a, b) => expr_uses_tensor(a) || expr_uses_tensor(b),
        ExprKind::JoinAll(items) => items.iter().any(expr_uses_tensor),
        ExprKind::MacroCall { args, .. } => args.iter().any(expr_uses_tensor),
    }
}

fn if_stmt_uses_tensor(s: &crate::ast::IfStmt) -> bool {
    s.branches
        .iter()
        .any(|(c, b)| expr_uses_tensor(c) || stmts_use_tensor(b))
        || s.else_body
            .as_ref()
            .map(|b| stmts_use_tensor(b))
            .unwrap_or(false)
}

fn match_stmt_uses_tensor(s: &crate::ast::MatchStmt) -> bool {
    expr_uses_tensor(&s.subject)
        || s.arms.iter().any(|a| {
            a.guard.as_ref().map(expr_uses_tensor).unwrap_or(false)
                || match &a.body {
                    MatchBody::Expr(e) => expr_uses_tensor(e),
                    MatchBody::Block(b) => stmts_use_tensor(b),
                }
        })
}

fn cond_clauses_use_tensor(clauses: &[CondClause]) -> bool {
    clauses.iter().any(|c| match c {
        CondClause::Let(_, e) => expr_uses_tensor(e),
        CondClause::LetPat(_, e) => expr_uses_tensor(e),
        CondClause::Expr(e) => expr_uses_tensor(e),
    })
}

pub(crate) fn stmts_use_tensor(stmts: &[Stmt]) -> bool {
    stmts.iter().any(stmt_uses_tensor)
}

fn stmt_uses_tensor(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Let(s) => s.value.as_ref().map(expr_uses_tensor).unwrap_or(false),
        Stmt::LetDestructure(s) => expr_uses_tensor(&s.value),
        Stmt::Return(s) => s.value.as_ref().map(expr_uses_tensor).unwrap_or(false),
        Stmt::Break(_, v) => v.as_ref().map(expr_uses_tensor).unwrap_or(false),
        Stmt::Continue(_) => false,
        Stmt::Throw(s) => s.value.as_ref().map(expr_uses_tensor).unwrap_or(false),
        Stmt::If(s) => if_stmt_uses_tensor(s),
        Stmt::IfLet(s) => {
            cond_clauses_use_tensor(&s.clauses)
                || stmts_use_tensor(&s.then_body)
                || s.elif_branches
                    .iter()
                    .any(|b| cond_clauses_use_tensor(&b.clauses) || stmts_use_tensor(&b.body))
                || s.else_body
                    .as_ref()
                    .map(|b| stmts_use_tensor(b))
                    .unwrap_or(false)
        }
        Stmt::Match(s) => match_stmt_uses_tensor(s),
        Stmt::While(s) => expr_uses_tensor(&s.condition) || stmts_use_tensor(&s.body),
        Stmt::WhileLet(s) => expr_uses_tensor(&s.value) || stmts_use_tensor(&s.body),
        Stmt::DoWhile(s) => stmts_use_tensor(&s.body) || expr_uses_tensor(&s.condition),
        Stmt::Loop(s) => stmts_use_tensor(&s.body),
        Stmt::Wait(e, _) => expr_uses_tensor(e),
        Stmt::For(s) => expr_uses_tensor(&s.iterable) || stmts_use_tensor(&s.body),
        Stmt::Guard(s) => {
            let cond_uses = match &s.cond {
                GuardCond::Expr(e) => expr_uses_tensor(e),
                GuardCond::Clauses(cs) => cond_clauses_use_tensor(cs),
            };
            cond_uses || stmts_use_tensor(&s.else_body)
        }
        Stmt::Try(s) => {
            stmts_use_tensor(&s.body) || s.catch_clauses.iter().any(|c| stmts_use_tensor(&c.body))
        }
        Stmt::Defer(body) => stmts_use_tensor(body),
        Stmt::Expr(e) => expr_uses_tensor(e),
        Stmt::Fn(f) => stmts_use_tensor(&f.body),
        Stmt::Struct(_) | Stmt::Enum(_) | Stmt::Mod(_) | Stmt::Alias(_) => false,
        Stmt::Yield(e, _) => expr_uses_tensor(e),
        Stmt::Comment(_) => false,
        Stmt::KernelBlock(s) => stmts_use_tensor(&s.body),
        Stmt::With(s) => stmts_use_tensor(&s.body),
    }
}
