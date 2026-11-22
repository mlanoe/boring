// Free helpers of the transpiler that analyse a program for GPU use: which parameters become
// device-resident or borrowed, which functions and structs touch a kernel, and the top-level
// `main` rename. Split out of `mod.rs`; its items are re-exported there so paths are unchanged.

use super::*;

/// Clone `program`, renaming any top-level, non-method function literally named
/// `main` to `new_name`. Used so the general transpiler doesn't try to emit its
/// own `fn main()`/`#[tokio::main]` wrapper for a boring program's entry point
/// when it's being embedded into a GPU target's own generated `main.rs`.
pub(crate) fn rename_top_level_main(program: &Program, new_name: &str) -> Program {
    let items = program.items.iter().map(|item| {
        if let Item::Fn(f) = item {
            if f.name == "main" && f.qualifier.is_none() {
                let mut renamed = f.clone();
                renamed.name = new_name.to_string();
                return Item::Fn(renamed);
            }
        }
        item.clone()
    }).collect();
    Program { items }
}

/// First `kernel` struct declaration found in `items`, recursing into `mod`
/// blocks (a kernel can be declared inside one). Used by `transpile_with_config`
/// to reject kernel usage on the `std` target before generating any Rust — see
/// its call site's doc comment for why. `Item::Kernel` presence is the complete
/// signal for "this program needs a GPU target": every `'unified`/`'global`/etc.
/// GPU memory qualifier only exists on a `KernelFieldDecl` (see `GpuQual`'s doc),
/// so there is no way to write GPU-callable code without declaring a `kernel`
/// struct first.
pub(super) fn first_kernel_decl(items: &[Item]) -> Option<&KernelDecl> {
    for item in items {
        match item {
            Item::Kernel(k) => return Some(k),
            Item::Mod(m) => {
                if let Some(k) = first_kernel_decl(&m.items) {
                    return Some(k);
                }
            }
            _ => {}
        }
    }
    None
}

/// True when `ty` (through `mut`) is a `'kernel'global`/`'kernel'unified` array type.
pub(crate) fn is_gpu_array_type(ty: &Type) -> bool {
    matches!(ty.without_mut(), Type::Qualified(inner, OwnerQual::GpuGlobal | OwnerQual::GpuUnified)
        if matches!(inner.without_mut(), Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) | Type::LabeledArray(_, _)))
}

/// True when `ty` (through `mut` and optional `'kernel'global`/`'kernel'unified`) is an array-like type.
pub(crate) fn is_array_like_type(ty: &Type) -> bool {
    let inner = match ty.without_mut() {
        Type::Qualified(inner, OwnerQual::GpuGlobal | OwnerQual::GpuUnified) => inner.without_mut(),
        other => other,
    };
    matches!(inner, Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) | Type::LabeledArray(_, _))
}

/// A parameter typed as a `'kernel'global`/`'kernel'unified` array (emitted by the host
/// emitter as a `BoringKernelArg<T>` parameter).
pub(crate) fn is_gpu_array_param(p: &Param) -> bool {
    p.ty.as_ref().is_some_and(is_gpu_array_type)
}

pub(super) fn struct_decl_gpu_fields(s: &crate::ast::StructDecl) -> Option<Vec<(String, bool)>> {
    if !s.inits.iter().all(|init| init.body.is_empty()) {
        return None;
    }
    let fields: Vec<(String, bool)> = if !s.fields.is_empty() {
        s.fields.iter().map(|fd| (fd.name.clone(), is_gpu_array_type(&fd.ty))).collect()
    } else if let Some(init) = s.inits.first() {
        init.params.iter()
            .map(|p| (p.name.clone(), p.ty.as_ref().is_some_and(is_gpu_array_type)))
            .collect()
    } else {
        Vec::new()
    };
    fields.iter().any(|(_, is_gpu)| *is_gpu).then_some(fields)
}

pub(super) fn gather_gpu_struct_fields_item(item: &Item, out: &mut std::collections::HashMap<String, Vec<(String, bool)>>) {
    match item {
        Item::Struct(s) => {
            if let Some(fields) = struct_decl_gpu_fields(s) {
                out.insert(s.name.clone(), fields);
            }
        }
        Item::Mod(m) => {
            for i in &m.items { gather_gpu_struct_fields_item(i, out); }
        }
        _ => {}
    }
}

pub(super) fn struct_ctor_takes_gpu_array(
    struct_fields: &std::collections::HashMap<String, Vec<(String, bool)>>,
    fn_name: &str,
    arg_idx: usize,
    label: Option<&str>,
) -> bool {
    let Some(fields) = struct_fields.get(fn_name) else { return false };
    match label {
        Some(l) => fields.iter().find(|(n, _)| n == l).is_some_and(|(_, is_gpu)| *is_gpu),
        None => fields.get(arg_idx).is_some_and(|(_, is_gpu)| *is_gpu),
    }
}

pub(super) fn kernel_ctor_takes_gpu_array(
    kernel_decls: &std::collections::HashMap<String, crate::ast::KernelDecl>,
    fn_name: &str,
    arg_idx: usize,
    label: Option<&str>,
) -> bool {
    let Some(decl) = kernel_decls.get(fn_name) else { return false };
    let Some(init) = decl.inits.first() else { return false };
    let init_param = match label {
        Some(l) => init.params.iter().find(|p| p.name == l),
        None => init.params.get(arg_idx),
    };
    let Some(init_param) = init_param else { return false };
    let Some(field_name) = kernel_init_field_for_param(decl, &init_param.name) else { return false };
    let Some(field) = decl.fields.iter().find(|fd| fd.name == field_name) else { return false };
    matches!(field.qual, crate::ast::GpuQual::Unified | crate::ast::GpuQual::Global)
        && matches!(field.ty, Type::Array(_) | Type::ArrayN(_, _))
}

pub(super) fn fn_param_index(all_fns: &[&FnDecl], fn_name: &str, arg_idx: usize, label: Option<&str>) -> Option<usize> {
    match label {
        Some(l) => all_fns.iter().find(|g| g.name == fn_name)
            .and_then(|g| g.params.iter().position(|p| p.name == l)),
        None => Some(arg_idx),
    }
}

fn expr_evaluates_to_var(expr: &Expr, name: &str) -> bool {
    match &expr.kind {
        ExprKind::Var(v) => v == name,
        ExprKind::If(if_stmt) => {
            let in_branches = if_stmt.branches.iter().any(|(_, b)| {
                b.last().is_some_and(|st| match st {
                    Stmt::Expr(e) => expr_evaluates_to_var(e, name),
                    _ => false,
                })
            });
            let in_else = if_stmt.else_body.as_ref().is_some_and(|b| {
                b.last().is_some_and(|st| match st {
                    Stmt::Expr(e) => expr_evaluates_to_var(e, name),
                    _ => false,
                })
            });
            in_branches || in_else
        }
        _ => false,
    }
}

/// True when `body` uses `name` in a way that expects or preserves `BoringKernelArg<T>`
/// beyond call-argument positions handled by `classify`: a `with <name>:` block, a
/// `'kernel'global`/`'kernel'unified` local initialized/assigned from `<name>`,
/// an assignment into a GPU-resident struct field, or a `kernel.tensors.*` readable operand.
pub(super) fn body_has_resident_use_of(
    body: &[Stmt],
    name: &str,
    struct_fields: &std::collections::HashMap<String, Vec<(String, bool)>>,
) -> bool {
    let mut resident_locals: std::collections::HashSet<String> = std::collections::HashSet::new();
    crate::ast::walk_stmts(body, &mut |node| {
        if let crate::ast::WalkNode::Stmt(Stmt::Let(l)) = node {
            if l.ty.as_ref().is_some_and(|t| t.without_mut().gpu_resident_qual().is_some()) {
                resident_locals.insert(l.name.clone());
            }
        }
    });
    let mut found = false;
    crate::ast::walk_stmts(body, &mut |node| {
        if found { return; }
        match node {
            crate::ast::WalkNode::Stmt(Stmt::With(w)) if w.names.iter().any(|n| n == name) => {
                found = true;
            }
            crate::ast::WalkNode::Stmt(Stmt::Let(l))
                if l.ty.as_ref().is_some_and(|t| t.without_mut().gpu_resident_qual().is_some())
                    && l.value.as_ref().is_some_and(|v| expr_evaluates_to_var(v, name)) =>
            {
                found = true;
            }
            crate::ast::WalkNode::Expr(Expr { kind: ExprKind::Assign(lhs, rhs), .. })
                if expr_evaluates_to_var(rhs, name) =>
            {
                if matches!(&lhs.kind, ExprKind::Var(ln) if resident_locals.contains(ln)) {
                    found = true;
                } else if let ExprKind::Field(_, field_name) = &lhs.kind {
                    if struct_fields.values().any(|fields| fields.iter().any(|(n, is_gpu)| n == field_name && *is_gpu)) {
                        found = true;
                    }
                }
            }
            crate::ast::WalkNode::Expr(Expr { kind: ExprKind::MethodCall(recv, method, args), .. })
                if matches!(method.as_str(), "matmul" | "mma" | "linear" | "moe_linear")
                    && matches!(&recv.kind, ExprKind::Field(gpu, namespace)
                        if namespace == "tensor" && matches!(&gpu.kind, ExprKind::Var(root) if root == "gpu")) =>
            {
                let positional = args.iter().position(|arg| arg.label.is_some()).unwrap_or(args.len());
                let readable_operands = if (method == "linear" && positional == 4) || method == "moe_linear" { 3 } else { 2 };
                if args.iter().take(readable_operands).any(|a| matches!(&a.value.kind, ExprKind::Var(v) if v == name)) {
                    found = true;
                }
            }
            _ => {}
        }
    });
    found
}

/// Free function name -> per-position flags: `true` for a `'kernel'global`/`'kernel'unified`
/// array parameter that every GPU backend passes as `&BoringKernelArg<T>` instead of by value.
/// `'kernel'global`/`'kernel'unified` describes memory residency, not ownership, so the borrow is
/// orthogonal to it: a by-value `BoringKernelArg<T>` would deep-copy the whole `Vec` (the `Host`
/// variant) at every call site for nothing. A parameter qualifies when it is not `mut`/`var`/variadic
/// and the body only reads it or forwards it to a kernel constructor's `'unified`/`'global` array
/// field or to another function's GPU array parameter (a size query is neutral) — returning,
/// storing, capturing or otherwise consuming it keeps it by value. Whole-program fixed point (a
/// forwarded parameter qualifies only when its use sites do), shared by the general pipeline and the
/// cuda/metal/rocm/wgpu host emitters so both sides of a call agree on the signature.
pub(crate) fn compute_gpu_borrow_params(
    program: &Program,
    kernel_decls: &std::collections::HashMap<String, crate::ast::KernelDecl>,
) -> std::collections::HashMap<String, Vec<bool>> {
    // The cuda/metal/rocm general pass has an empty `gpu_kernels`, so also take the kernels the
    // program itself declares (inline `kernel(...)` blocks are lowered to ones).
    let mut kernel_decls = kernel_decls.clone();
    for item in &program.items {
        if let Item::Kernel(k) = item { kernel_decls.entry(k.name.clone()).or_insert_with(|| k.clone()); }
    }
    let mut struct_fields: std::collections::HashMap<String, Vec<(String, bool)>> = std::collections::HashMap::new();
    for item in &program.items {
        gather_gpu_struct_fields_item(item, &mut struct_fields);
    }
    let mut all_fns: Vec<&FnDecl> = Vec::new();
    for item in &program.items { Transpiler::gather_gpu_arg_fns_item(item, &mut all_fns); }
    // The shared entry has only one overload's arity, so skip every overload when iterating.
    let overloaded: std::collections::HashSet<&str> = all_fns.iter()
        .filter(|f| all_fns.iter().filter(|g| g.name == f.name).count() > 1)
        .map(|f| f.name.as_str()).collect();
    // Least fixed point: which `(fn_name, param_idx)` actually take `BoringKernelArg<T>`?
    // A plain host/device helper (`[uint8]'global buf` indexing `buf[i]`, or a wrapper around one)
    // stays `&Vec<T>` in the general pass and must never be treated as a `BoringKernelArg` target.
    let mut takes_kernel_arg: std::collections::HashMap<&str, Vec<bool>> = all_fns.iter()
        .map(|f| {
            let ret_res = fn_returns_resident(f);
            let row = f.params.iter().map(|p| {
                is_gpu_array_param(p) && (ret_res || body_has_resident_use_of(&f.body, &p.name, &struct_fields))
            }).collect();
            (f.name.as_str(), row)
        })
        .collect();
    loop {
        let mut changed = false;
        for f in &all_fns {
            if overloaded.contains(f.name.as_str()) { continue; }
            for (i, p) in f.params.iter().enumerate() {
                if takes_kernel_arg[f.name.as_str()][i] || !is_gpu_array_param(p) { continue; }
                let mut has_qualifying_call = false;
                let mut classify = |fn_name: &str, arg_idx: usize, label: Option<&str>| -> bool {
                    let ok = kernel_ctor_takes_gpu_array(&kernel_decls, fn_name, arg_idx, label)
                        || struct_ctor_takes_gpu_array(&struct_fields, fn_name, arg_idx, label)
                        || fn_param_index(&all_fns, fn_name, arg_idx, label)
                            .and_then(|idx| takes_kernel_arg.get(fn_name).and_then(|v| v.get(idx).copied()))
                            .unwrap_or(false);
                    if ok { has_qualifying_call = true; }
                    ok
                };
                let treat_len = p.ty.as_ref().is_some_and(is_array_like_type);
                let (any, only) = crate::ast::scan_var_call_arg_uses(&f.body, &p.name, &mut classify, treat_len);
                if has_qualifying_call || !any || only {
                    takes_kernel_arg.get_mut(f.name.as_str()).unwrap()[i] = true;
                    changed = true;
                }
            }
        }
        if !changed { break; }
    }
    let candidate = |fn_name: &str, i: usize, p: &Param| {
        !p.mutable && !p.rebindable && !p.variadic && p.default.is_none()
            && is_gpu_array_param(p)
            && takes_kernel_arg.get(fn_name).and_then(|v| v.get(i).copied()).unwrap_or(false)
    };
    let mut flags: std::collections::HashMap<&str, Vec<bool>> = all_fns.iter()
        .map(|f| (f.name.as_str(), f.params.iter().enumerate().map(|(i, p)| candidate(f.name.as_str(), i, p)).collect()))
        .collect();
    // Overloaded names would alias one entry: leave them by value.
    for f in &all_fns {
        if overloaded.contains(f.name.as_str()) {
            if let Some(v) = flags.get_mut(f.name.as_str()) { v.iter_mut().for_each(|b| *b = false); }
        }
    }
    // Greatest fixed point: start optimistic, drop a position as soon as a use disqualifies it.
    loop {
        let mut changed = false;
        for f in &all_fns {
            if overloaded.contains(f.name.as_str()) { continue; }
            for (i, p) in f.params.iter().enumerate() {
                if !flags[f.name.as_str()][i] { continue; }
                let mut classify = |fn_name: &str, arg_idx: usize, label: Option<&str>| -> bool {
                    if kernel_ctor_takes_gpu_array(&kernel_decls, fn_name, arg_idx, label) {
                        return true;
                    }
                    // Storing into a struct field (`Plane(x, ...)`) consumes an owned `BoringKernelArg<T>`,
                    // so only forwarding to another function keeps the parameter borrowed.
                    fn_param_index(&all_fns, fn_name, arg_idx, label)
                        .and_then(|idx| takes_kernel_arg.get(fn_name).and_then(|v| v.get(idx).copied()))
                        .unwrap_or(false)
                };
                let (any, only) = crate::ast::scan_var_call_arg_uses(&f.body, &p.name, &mut classify, true);
                if !any || !only {
                    flags.get_mut(f.name.as_str()).unwrap()[i] = false;
                    changed = true;
                }
            }
        }
        if !changed { break; }
    }
    flags.into_iter()
        .filter(|(_, v)| v.iter().any(|b| *b))
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

pub(super) fn fn_returns_resident(g: &FnDecl) -> bool {
    g.return_ty.as_ref().is_some_and(|ty| match ty {
        Type::Tuple(elems) => elems.iter().any(|t| t.gpu_resident_qual().is_some()),
        other => other.gpu_resident_qual().is_some(),
    })
}

/// Names of the kernel-touching free functions whose arguments the host emitter must wrap.
pub(super) fn host_wrapped_fn_names(program: &Program, kernel_touching: &std::collections::HashSet<String>) -> std::collections::HashSet<String> {
    program.items.iter().filter_map(|it| match it {
        Item::Fn(g) if kernel_touching.contains(&g.name) && g.params.iter().any(is_gpu_array_param) && !fn_returns_resident(g) => Some(g.name.clone()),
        _ => None,
    }).collect()
}

/// Names of free functions whose body directly constructs a `kernel` instance
/// (`KernelName(...)`) or contains a `kernel: ...` dispatch block. These are the
/// only functions cuda/metal can't hand off to the general pipeline (see this
/// section's header comment) — everything else, including functions that merely
/// *call* one of these (e.g. a struct method that calls `linear_gpu(...)`), is
/// unaffected and goes through the general pipeline normally.
///
/// Scope: only free (`Item::Fn`) functions are checked. A `kernel`-touching
/// struct *method* is not detected here — see `strip_top_level_fns`'s doc for
/// why that combination isn't supported and is a documented gap, not silently
/// mishandled (callers should check for it separately if they care).
pub(crate) fn kernel_touching_fn_names(program: &Program, kernel_names: &std::collections::HashSet<String>) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            let param_is_kernel = f.params.iter().any(|p| matches!(&p.ty, Some(Type::Named(n)) if kernel_names.contains(n)));
            if param_is_kernel || stmts_touch_kernel(&f.body, kernel_names) {
                out.insert(f.name.clone());
            }
        }
    }
    // A kernel-touching function (that doesn't return a resident value) with a
    // `'kernel'global`/`'kernel'unified` array parameter is emitted by the backend's own host emitter with a
    // `BoringKernelArg<T>` parameter (dual host/resident). A caller emitted by the
    // *general* pipeline has no idea about that signature and passes a plain
    // `Vec<T>` (rustc E0308), so every free function that calls such a function
    // is kernel-touching too (transitively), keeping the whole call chain on the
    // host emitter, which wraps/forwards the argument correctly.
    // (A callee that returns a resident value is already handled by the general
    // pipeline: `compute_gpu_arg_params` flags its qualified parameters, so its
    // callers wrap/forward the argument themselves.)
    loop {
        let mut changed = false;
        for item in &program.items {
            let Item::Fn(f) = item else { continue };
            if out.contains(&f.name) { continue; }
            let calls_resident_taking_fn = {
                let mut found = false;
                crate::ast::walk_stmts(&f.body, &mut |node| {
                    if let crate::ast::WalkNode::Expr(Expr { kind: ExprKind::Call(callee, _), .. }) = node {
                        if let ExprKind::Var(name) = &callee.kind {
                            if out.contains(name) && program.items.iter().any(|it| matches!(it,
                                Item::Fn(g) if g.name == *name && g.params.iter().any(is_gpu_array_param) && !fn_returns_resident(g)))
                            {
                                found = true;
                            }
                        }
                    }
                });
                found
            };
            if calls_resident_taking_fn {
                out.insert(f.name.clone());
                changed = true;
            }
        }
        if !changed { break; }
    }
    out
}

/// True when any method of any `Item::Struct` in `program` is kernel-touching
/// (see `kernel_touching_fn_names`'s doc for the definition). `cuda`/`metal`'s
/// `transpile_*` entry points check this and, if true, do NOT attempt the
/// splice architecture for the affected struct at all (falling back to fully
/// custom emission for it) — routing such a struct through the general
/// pipeline while ALSO custom-emitting it for its kernel-touching method would
/// double-define the struct/impl block. Not exercised by this codebase's own
/// `.br` corpus (every kernel-touching construct here is a free function), so
/// this exists purely as a safety check, not a fully-engineered dual path.
pub(crate) fn kernel_touching_struct_names(program: &Program, kernel_names: &std::collections::HashSet<String>) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for item in &program.items {
        if let Item::Struct(s) = item {
            if s.methods.iter().any(|m| stmts_touch_kernel(&m.body, kernel_names)) {
                out.insert(s.name.clone());
            }
        }
    }
    out
}

/// True when the program's TOP-LEVEL statements/lets (as opposed to inside a
/// named function — see `kernel_touching_fn_names`) directly construct a
/// `kernel` instance or contain a `kernel: ...` dispatch block. Common in
/// practice for GPU-target example programs with no `def main():` at all
/// (e.g. `examples/vector_add_gpu.br`: `mut k = VectorAdd(...); kernel:
/// k(block=256)`, both bare top-level statements) — cuda/metal's
/// `transpile_*` entry points check this and, if true, keep top-level
/// statement/let handling entirely on their own custom emitter (matching
/// their pre-splice behavior, which already worked for this case) instead of
/// letting the general pipeline fold it into a synthesized `boring_main` (see
/// `TranspileConfig::gpu_top_level_handled_by_host`) — the general pipeline's
/// kernel-aware codegen for that fold-in is wgpu-shaped and can't run with
/// `gpu_kernels` empty (see `cuda::mod`'s doc comment for the full story).
///
/// A top-level call to a kernel-touching free function (`kernel_touching`, from
/// `kernel_touching_fn_names`) counts too, bare or as a `let` initializer or nested
/// in an `if`/`for` block, when it takes a `'kernel'global`/`'kernel'unified` array
/// parameter: that function is emitted by the host emitter with `BoringKernelArg<T>`
/// parameters, and only the host emitter wraps the arguments. (A call with no such
/// argument stays on the general pass, which promotes top-level arrays to statics.)
pub(crate) fn top_level_touches_kernel(
    program: &Program,
    kernel_names: &std::collections::HashSet<String>,
    kernel_touching: &std::collections::HashSet<String>,
) -> bool {
    let kernel_touching = &host_wrapped_fn_names(program, kernel_touching);
    let calls_kernel_touching = |found: &mut bool, node: crate::ast::WalkNode| {
        if let crate::ast::WalkNode::Expr(Expr { kind: ExprKind::Call(callee, _), .. }) = node {
            if matches!(&callee.kind, ExprKind::Var(n) if kernel_touching.contains(n)) {
                *found = true;
            }
        }
    };
    for item in &program.items {
        match item {
            Item::Let(s)
                if s.value.as_ref().is_some_and(|v| expr_constructs_kernel(v, kernel_names)) =>
            {
                return true;
            }
            Item::Let(s) if !kernel_touching.is_empty() && s.value.as_ref().is_some_and(|v| {
                let mut found = false;
                crate::ast::walk_expr(v, &mut |n| calls_kernel_touching(&mut found, n));
                found
            }) => {
                return true;
            }
            Item::Stmt(s) if stmt_touches_kernel(s, kernel_names) => {
                return true;
            }
            Item::Stmt(s) if !kernel_touching.is_empty() && {
                let mut found = false;
                crate::ast::walk_stmts(std::slice::from_ref(s), &mut |n| calls_kernel_touching(&mut found, n));
                found
            } => {
                return true;
            }
            _ => {}
        }
    }
    false
}


pub(super) fn expr_constructs_kernel(e: &Expr, kernel_names: &std::collections::HashSet<String>) -> bool {
    match &e.kind {
        ExprKind::Call(callee, _) => matches!(&callee.kind, ExprKind::Var(n) if kernel_names.contains(n.as_str())),
        _ => false,
    }
}


pub(super) fn stmts_touch_kernel(body: &[Stmt], kernel_names: &std::collections::HashSet<String>) -> bool {
    body.iter().any(|s| stmt_touches_kernel(s, kernel_names))
}

pub(super) fn stmt_touches_kernel(stmt: &Stmt, kernel_names: &std::collections::HashSet<String>) -> bool {
    match stmt {
        Stmt::KernelBlock(_) => true,
        Stmt::Let(s) => s.value.as_ref().is_some_and(|v| expr_constructs_kernel(v, kernel_names)),
        Stmt::LetDestructure(s) => expr_constructs_kernel(&s.value, kernel_names),
        Stmt::If(i) => i.branches.iter().any(|(_, b)| stmts_touch_kernel(b, kernel_names))
            || i.else_body.as_ref().is_some_and(|b| stmts_touch_kernel(b, kernel_names)),
        Stmt::While(w) => stmts_touch_kernel(&w.body, kernel_names),
        Stmt::DoWhile(w) => stmts_touch_kernel(&w.body, kernel_names),
        Stmt::For(f) => stmts_touch_kernel(&f.body, kernel_names),
        Stmt::Loop(l) => stmts_touch_kernel(&l.body, kernel_names),
        Stmt::Guard(g) => stmts_touch_kernel(&g.else_body, kernel_names),
        Stmt::With(w) => stmts_touch_kernel(&w.body, kernel_names),
        Stmt::Defer(b) => stmts_touch_kernel(b, kernel_names),
        Stmt::Try(t) => stmts_touch_kernel(&t.body, kernel_names)
            || t.catch_clauses.iter().any(|c| stmts_touch_kernel(&c.body, kernel_names)),
        Stmt::Match(m) => m.arms.iter().any(|a| match &a.body {
            MatchBody::Block(b) => stmts_touch_kernel(b, kernel_names),
            MatchBody::Expr(e) => expr_constructs_kernel(e, kernel_names),
        }),
        _ => false,
    }
}

/// Remove each top-level `pub fn {name}(...) { ... }` / `fn {name}(...) { ... }`
/// definition (matched by brace-depth, not just text search) from `code`, for
/// every `name` in `names`.
///
/// Used to discard the general pipeline's rendering of a kernel-touching
/// function's BODY (which is necessarily wrong for cuda/metal — see this
/// section's header comment: the pipeline emits wgpu-shaped kernel construction/
/// dispatch calls that don't exist on cuda/metal's real kernel structs), while
/// still routing the function through the general pipeline for the purpose of
/// registering its signature/`throws`-ness — every OTHER (non-kernel-touching)
/// function's calls TO `name` still went through the general pipeline's own
/// call-site emission (by-ref array/dict/set/struct coercion, `?`-propagation,
/// etc.), which needed `name`'s real declaration to get that right. The GPU
/// backend's own custom emitter supplies the real (native, fallible-dispatch)
/// replacement body for each stripped name separately, with a matching by-ref
/// signature (see `cuda::host`'s `fn_ref_params`).
///
/// Assumes (true of every function this transpiler emits, checked against every
/// sample in this codebase) that a function signature is never wrapped across
/// multiple lines — matching is done against a single trimmed line's prefix.
pub(crate) fn strip_top_level_fns(code: &str, names: &std::collections::HashSet<String>) -> String {
    if names.is_empty() { return code.to_string(); }
    let mut out = String::with_capacity(code.len());
    let mut i = 0;
    while i < code.len() {
        let line_end = code[i..].find('\n').map(|p| i + p + 1).unwrap_or(code.len());
        let line = &code[i..line_end];
        let trimmed = line.trim_start();
        let matched = names.iter().any(|n| {
            trimmed.starts_with(&format!("fn {}(", n)) || trimmed.starts_with(&format!("pub fn {}(", n))
        });
        if matched {
            let rest = &code[i..];
            let mut depth = 0i32;
            let mut seen_open = false;
            let mut end_offset = rest.len();
            for (off, ch) in rest.char_indices() {
                if ch == '{' { depth += 1; seen_open = true; }
                if ch == '}' {
                    depth -= 1;
                    if seen_open && depth == 0 {
                        let after = &rest[off + 1..];
                        end_offset = after.find('\n').map(|p| off + 1 + p + 1).unwrap_or(rest.len());
                        break;
                    }
                }
            }
            i += end_offset;
            continue;
        }
        out.push_str(line);
        i = line_end;
    }
    out
}

/// Given the renamed program (see `rename_top_level_main`) and the general
/// pipeline's output for it, determine whether a `boring_main` exists and
/// whether it returns a `Result` — shared by wgpu/cuda/metal so each backend's
/// own real `fn main()` knows whether/how to call it. See `wgpu::transpile_wgpu`
/// for the original derivation of this logic (this is a verbatim extraction).
pub(crate) fn detect_boring_main(renamed_program: &Program, general_out: &TranspileOutput) -> (bool, bool) {
    let explicit_boring_main_throws = renamed_program.items.iter().find_map(|item| {
        match item {
            Item::Fn(f) if f.name == "boring_main" && f.qualifier.is_none() => Some(f.throws),
            _ => None,
        }
    });
    let has_boring_main = general_out.gpu_main_emitted || explicit_boring_main_throws.is_some();
    let boring_main_throws = explicit_boring_main_throws.unwrap_or(true);
    (has_boring_main, boring_main_throws)
}

