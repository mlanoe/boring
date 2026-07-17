// GPU kernel interpreter — simulation mode.
//
// In simulation mode (`boring run` or default), GPU kernels execute sequentially
// on the CPU. The `kernel(...)` expression runs the anonymous `def ()` entry point
// in a loop over all threads, then wraps the result in a `KernelHandle` whose
// `.wait()` immediately returns it and `.done()` returns true.
//
// GPU builtins (`gpu.thread.x`, `gpu.block.x`, etc.) are injected into the
// execution environment before each thread iteration. `sync` is a no-op.

use super::*;
use std::rc::Rc;

impl Interpreter {
    /// Register a `kernel Name:` declaration in the environment.
    /// The kernel struct is stored as a `Value::KernelStruct` callable.
    pub(crate) fn exec_kernel_decl(&mut self, decl: &crate::ast::KernelDecl, env: EnvRef) -> Result<(), Signal> {
        let val = Value::KernelStruct { decl: decl.clone(), captured: Rc::clone(&env) };
        env.borrow_mut().define(&decl.name, val.clone());
        if !Rc::ptr_eq(&env, &self.global) {
            self.global.borrow_mut().define(&decl.name, val);
        }
        Ok(())
    }

    /// Evaluate a `kernel(block = N, ...) expr` expression.
    ///
    /// Simulation strategy:
    /// - Evaluate `expr` to obtain the kernel instance (an Object with fields).
    /// - Look up the anonymous `def ()` method in the kernel's declaration.
    /// - Determine thread/block dimensions from `config.block` (default: 1).
    /// - Run the entry point once per thread, injecting `gpu.*` builtins.
    /// - Return a `KernelHandle { result: kernel_instance }`.
    pub(crate) fn eval_kernel_launch(
        &mut self,
        config: &crate::ast::KernelConfig,
        kernel_expr: &Expr,
        env: EnvRef,
    ) -> Eval {
        let _line = kernel_expr.line;

        // Evaluate the kernel instance.
        let kernel_val = self.eval_expr(kernel_expr, Rc::clone(&env))?;

        // Extract the Object fields and the KernelDecl we need.
        let (type_name, fields, decl, captured) = match &kernel_val {
            Value::Object(inner) => {
                let type_name = inner.borrow().type_name.clone();
                let fields = inner.borrow().fields.clone();
                // Look up the kernel declaration from the environment.
                let kval = env.borrow().get(&type_name);
                match kval {
                    Some(Value::KernelStruct { decl, captured }) => {
                        (type_name, fields, decl, captured)
                    }
                    _ => {
                        // Fallback: no declaration found — return handle immediately.
                        return Ok(Value::KernelHandle { result: Box::new(kernel_val) });
                    }
                }
            }
            _ => {
                return Ok(Value::KernelHandle { result: Box::new(kernel_val) });
            }
        };

        // Find the anonymous entry-point method: `def ()` has name "".
        let entry_point = decl.methods.iter().find(|m| m.name.is_empty() && m.params.is_empty());

        if let Some(entry) = entry_point {
            // Determine block size (1D or 2D).
            let (block_x, block_y) = if let Some(block_expr) = &config.block {
                match self.eval_expr(block_expr, Rc::clone(&env))? {
                    Value::Int(n) => (n.max(1) as usize, 1),
                    Value::Tuple(t) => {
                        let x = match t.first() { Some(Value::Int(n)) => (*n).max(1) as usize, _ => 1 };
                        let y = match t.get(1) { Some(Value::Int(n)) => (*n).max(1) as usize, _ => 1 };
                        (x, y)
                    }
                    _ => (1, 1),
                }
            } else {
                (1, 1)
            };

            // Determine grid size (1D or 2D).
            let (grid_x, grid_y) = if let Some(grid_expr) = &config.grid {
                match self.eval_expr(grid_expr, Rc::clone(&env))? {
                    Value::Int(n) => (n.max(1) as usize, 1),
                    Value::Tuple(t) => {
                        let x = match t.first() { Some(Value::Int(n)) => (*n).max(1) as usize, _ => 1 };
                        let y = match t.get(1) { Some(Value::Int(n)) => (*n).max(1) as usize, _ => 1 };
                        (x, y)
                    }
                    _ => (1, 1),
                }
            } else {
                // Infer 1D grid from the longest array field; Y defaults to 1.
                let max_len = fields.iter()
                    .filter_map(|(_, v)| if let Value::Array(a) = v { Some(a.len()) } else { None })
                    .max()
                    .unwrap_or(0);
                let inferred_x = if max_len > 0 { max_len.div_ceil(block_x * block_y) } else { 1 };
                (inferred_x, 1)
            };

            let total_threads = block_x * block_y * grid_x * grid_y;

            // Build the kernel object to mutate during simulation.
            let kernel_obj = make_object(type_name.clone(), fields);

            // Run the entry point for each thread sequentially (row-major: bx, by, tx, ty).
            for thread_idx in 0..total_threads {
                // Decompose flat index into 2D block + 2D thread coordinates.
                let threads_per_block = block_x * block_y;
                let flat_block = thread_idx / threads_per_block;
                let flat_thread = thread_idx % threads_per_block;

                let block_idx_x = flat_block % grid_x;
                let block_idx_y = flat_block / grid_x;
                let thread_in_block_x = flat_thread % block_x;
                let thread_in_block_y = flat_thread / block_x;

                // Create a child environment for this thread.
                let thread_env = Env::child(Rc::clone(&captured));

                // Bind all kernel fields into the thread environment.
                // Read from the CURRENT kernel_obj so that sequential thread writes accumulate.
                if let Value::Object(ref obj) = kernel_obj {
                    let current_fields = obj.borrow().fields.clone();
                    for (field_decl, (name, val)) in decl.fields.iter().zip(current_fields.iter()) {
                        match field_decl.binding {
                            FieldBinding::Mut | FieldBinding::Var =>
                                thread_env.borrow_mut().define_mut(name, val.clone()),
                            FieldBinding::Let =>
                                thread_env.borrow_mut().define(name, val.clone()),
                        }
                    }
                    // Also bind `self` so methods can call each other.
                    thread_env.borrow_mut().define("self", kernel_obj.clone());
                }
                // Remember the initial values of mutable fields so we can detect writes.
                let initial_vals: Vec<(String, Value)> = if let Value::Object(ref obj) = kernel_obj {
                    decl.fields.iter()
                        .filter(|f| matches!(f.binding, FieldBinding::Mut | FieldBinding::Var))
                        .filter_map(|f| {
                            let v = obj.borrow().fields.iter().find(|(n, _)| n == &f.name).map(|(_, v)| v.clone());
                            v.map(|v| (f.name.clone(), v))
                        })
                        .collect()
                } else { vec![] };

                // Inject gpu.* builtins as a nested object.
                let gpu_thread = make_object("GpuThread".into(), vec![
                    ("x".into(), Value::Int(thread_in_block_x as i64)),
                    ("y".into(), Value::Int(thread_in_block_y as i64)),
                    ("z".into(), Value::Int(0)),
                ]);
                let gpu_block = make_object("GpuBlock".into(), vec![
                    ("x".into(), Value::Int(block_idx_x as i64)),
                    ("y".into(), Value::Int(block_idx_y as i64)),
                    ("z".into(), Value::Int(0)),
                ]);
                let gpu_block_dim = make_object("GpuBlockDim".into(), vec![
                    ("x".into(), Value::Int(block_x as i64)),
                    ("y".into(), Value::Int(block_y as i64)),
                    ("z".into(), Value::Int(1)),
                ]);
                let gpu_grid_dim = make_object("GpuGridDim".into(), vec![
                    ("x".into(), Value::Int(grid_x as i64)),
                    ("y".into(), Value::Int(grid_y as i64)),
                    ("z".into(), Value::Int(1)),
                ]);
                let gpu_val = make_object("Gpu".into(), vec![
                    ("thread".into(), gpu_thread),
                    ("block".into(), gpu_block),
                    ("block_dim".into(), gpu_block_dim),
                    ("grid_dim".into(), gpu_grid_dim),
                ]);
                thread_env.borrow_mut().define("gpu", gpu_val);

                // Bind all kernel methods as functions in the thread environment.
                for method in &decl.methods {
                    if !method.name.is_empty() {
                        let fn_val = Value::Fn { decl: method.clone(), captured: Rc::clone(&thread_env) };
                        thread_env.borrow_mut().define(&method.name, fn_val);
                    }
                }

                // Execute the entry point body.
                let result = self.exec_block(&entry.body, Rc::clone(&thread_env));
                match result {
                    Ok(_) | Err(Signal::Return(_)) => {
                        // Write back mutable fields that changed during this thread's execution.
                        // Only write back fields whose value differs from what this thread started with,
                        // to avoid threads that did NOT touch a field overwriting an earlier thread's update.
                        if let Value::Object(ref obj) = kernel_obj {
                            for (field_name, init_val) in &initial_vals {
                                let new_val = thread_env.borrow().get(field_name);
                                if let Some(new_val) = new_val {
                                    let changed = match (&new_val, init_val) {
                                        (Value::Array(a), Value::Array(b)) => a != b,
                                        _ => new_val != *init_val,
                                    };
                                    if changed {
                                        let mut o = obj.borrow_mut();
                                        if let Some(entry) = o.fields.iter_mut().find(|(n, _)| n == field_name) {
                                            entry.1 = new_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => return Err(e),
                }
            }

            Ok(Value::KernelHandle { result: Box::new(kernel_obj) })
        } else {
            // No entry point — return handle immediately.
            Ok(Value::KernelHandle { result: Box::new(make_object(type_name, fields.clone())) })
        }
    }

    /// `k(block = N)` short-hand dispatch.
    /// Called by `eval_expr` when a kernel Object is invoked with a `block =` label.
    /// `block_val` is the already-evaluated block dimension (Int or Tuple).
    pub(crate) fn eval_kernel_launch_with_val(
        &mut self,
        _config: crate::ast::KernelConfig,
        kernel_val: Value,
        block_val: Value,
        line: usize,
        env: &EnvRef,
    ) -> Eval {
        let (type_name, fields, decl, captured) = match &kernel_val {
            Value::Object(inner) => {
                let type_name = inner.borrow().type_name.clone();
                let fields = inner.borrow().fields.clone();
                let kval = env.borrow().get(&type_name);
                match kval {
                    Some(Value::KernelStruct { decl, captured }) => (type_name, fields, decl, captured),
                    _ => return Ok(Value::KernelHandle { result: Box::new(kernel_val) }),
                }
            }
            _ => return Ok(Value::KernelHandle { result: Box::new(kernel_val) }),
        };

        let (block_x, block_y) = match block_val {
            Value::Int(n)  => (n.max(1) as usize, 1),
            Value::Uint(n) => (n.max(1) as usize, 1),
            Value::Tuple(ref t) => {
                let x = match t.first() {
                    Some(Value::Int(n))  => (*n).max(1) as usize,
                    Some(Value::Uint(n)) => (*n).max(1) as usize,
                    _ => 1,
                };
                let y = match t.get(1) {
                    Some(Value::Int(n))  => (*n).max(1) as usize,
                    Some(Value::Uint(n)) => (*n).max(1) as usize,
                    _ => 1,
                };
                (x, y)
            }
            _ => (1, 1),
        };

        // Infer grid from the longest array field.
        let max_len = fields.iter()
            .filter_map(|(_, v)| if let Value::Array(a) = v { Some(a.len()) } else { None })
            .max()
            .unwrap_or(0);
        let (grid_x, grid_y) = if max_len > 0 {
            let bxy = block_x * block_y;
            let total_threads = max_len;
            // For 2D kernels infer a square-ish grid from dim fields if available.
            let dim_w = fields.iter().find(|(n, _)| n == "dim")
                .and_then(|(_, v)| if let Value::Object(o) = v {
                    o.borrow().fields.iter().find(|(k, _)| k == "width").and_then(|(_, w)| {
                        if let Value::Uint(n) = w { Some(*n as usize) } else { None }
                    })
                } else { None });
            let dim_h = fields.iter().find(|(n, _)| n == "dim")
                .and_then(|(_, v)| if let Value::Object(o) = v {
                    o.borrow().fields.iter().find(|(k, _)| k == "height").and_then(|(_, h)| {
                        if let Value::Uint(n) = h { Some(*n as usize) } else { None }
                    })
                } else { None });
            if let (Some(w), Some(h)) = (dim_w, dim_h) {
                let gx = w.div_ceil(block_x);
                let gy = h.div_ceil(block_y);
                (gx, gy)
            } else {
                let gx = total_threads.div_ceil(bxy);
                (gx, 1)
            }
        } else {
            (1, 1)
        };

        let entry_point = decl.methods.iter().find(|m| m.name.is_empty() && m.params.is_empty());

        if let Some(entry) = entry_point {
            let total_threads = block_x * block_y * grid_x * grid_y;
            let kernel_obj = make_object(type_name.clone(), fields);

            for thread_idx in 0..total_threads {
                let threads_per_block = block_x * block_y;
                let flat_block = thread_idx / threads_per_block;
                let flat_thread = thread_idx % threads_per_block;
                let block_idx_x = flat_block % grid_x;
                let block_idx_y = flat_block / grid_x;
                let thread_in_block_x = flat_thread % block_x;
                let thread_in_block_y = flat_thread / block_x;

                let thread_env = Env::child(Rc::clone(&captured));

                if let Value::Object(ref obj) = kernel_obj {
                    let current_fields = obj.borrow().fields.clone();
                    for (field_decl, (name, val)) in decl.fields.iter().zip(current_fields.iter()) {
                        match field_decl.binding {
                            FieldBinding::Mut | FieldBinding::Var =>
                                thread_env.borrow_mut().define_mut(name, val.clone()),
                            FieldBinding::Let =>
                                thread_env.borrow_mut().define(name, val.clone()),
                        }
                    }
                    thread_env.borrow_mut().define("self", kernel_obj.clone());
                }

                let initial_vals: Vec<(String, Value)> = if let Value::Object(ref obj) = kernel_obj {
                    decl.fields.iter()
                        .filter(|f| matches!(f.binding, FieldBinding::Mut | FieldBinding::Var))
                        .filter_map(|f| {
                            obj.borrow().fields.iter().find(|(n, _)| n == &f.name).map(|(_, v)| (f.name.clone(), v.clone()))
                        })
                        .collect()
                } else { vec![] };

                let gpu_thread   = make_object("GpuThread".into(),   vec![("x".into(), Value::Int(thread_in_block_x as i64)), ("y".into(), Value::Int(thread_in_block_y as i64)), ("z".into(), Value::Int(0))]);
                let gpu_block    = make_object("GpuBlock".into(),    vec![("x".into(), Value::Int(block_idx_x as i64)),       ("y".into(), Value::Int(block_idx_y as i64)),       ("z".into(), Value::Int(0))]);
                let gpu_block_dim = make_object("GpuBlockDim".into(), vec![("x".into(), Value::Int(block_x as i64)),          ("y".into(), Value::Int(block_y as i64)),          ("z".into(), Value::Int(1))]);
                let gpu_grid_dim  = make_object("GpuGridDim".into(),  vec![("x".into(), Value::Int(grid_x as i64)),           ("y".into(), Value::Int(grid_y as i64)),           ("z".into(), Value::Int(1))]);
                let gpu_val = make_object("Gpu".into(), vec![
                    ("thread".into(), gpu_thread), ("block".into(), gpu_block),
                    ("block_dim".into(), gpu_block_dim), ("grid_dim".into(), gpu_grid_dim),
                ]);
                thread_env.borrow_mut().define("gpu", gpu_val);

                for method in &decl.methods {
                    if !method.name.is_empty() {
                        let fn_val = Value::Fn { decl: method.clone(), captured: Rc::clone(&thread_env) };
                        thread_env.borrow_mut().define(&method.name, fn_val);
                    }
                }

                let result = self.exec_block(&entry.body, Rc::clone(&thread_env));
                match result {
                    Ok(_) | Err(Signal::Return(_)) => {
                        if let Value::Object(ref obj) = kernel_obj {
                            for (field_name, init_val) in &initial_vals {
                                let new_val = thread_env.borrow().get(field_name);
                                if let Some(new_val) = new_val {
                                    let changed = match (&new_val, init_val) {
                                        (Value::Array(a), Value::Array(b)) => a != b,
                                        _ => new_val != *init_val,
                                    };
                                    if changed {
                                        let mut o = obj.borrow_mut();
                                        if let Some(entry) = o.fields.iter_mut().find(|(n, _)| n == field_name) {
                                            entry.1 = new_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => return Err(e),
                }
            }

            // Write back updated fields to the environment binding for `k`.
            // `k(block=N)` should update `k` in place (rebind after wait).
            let _ = line; // used in error paths only
            Ok(Value::KernelHandle { result: Box::new(kernel_obj) })
        } else {
            Ok(Value::KernelHandle { result: Box::new(make_object(type_name, fields.clone())) })
        }
    }

    /// Instantiate a kernel struct: call its `init` declaration (if any),
    /// or construct a plain Object from the fields.
    pub(crate) fn instantiate_kernel_struct(
        &mut self,
        decl: &crate::ast::KernelDecl,
        captured: &EnvRef,
        args: Vec<Value>,
        _line: usize,
    ) -> Eval {
        // Build initial field values (Nil for each field).
        let mut fields: Vec<(String, Value)> = decl.fields
            .iter()
            .map(|f| (f.name.clone(), Value::Nil))
            .collect();

        // Select the matching init overload. Strategy:
        // 1. Exact type match: first param's declared type matches the runtime value's type name.
        // 2. Arity match: same number of params as args.
        // 3. First init as fallback.
        let init_decl = decl.inits.iter()
            .find(|i| {
                if i.params.len() != args.len() { return false; }
                // Check if first param type annotation matches first arg's type name.
                if let (Some(first_param), Some(first_arg)) = (i.params.first(), args.first()) {
                    let expected = first_param.ty.as_ref().map(|t| match t {
                        crate::ast::Type::Named(n) => n.as_str(),
                        _ => "",
                    });
                    let actual = first_arg.type_name();
                    // Type annotation matches runtime type → this is the right overload.
                    if let Some(exp) = expected {
                        if !exp.is_empty() && exp != actual.as_str() { return false; }
                    }
                }
                true
            })
            .or_else(|| decl.inits.first());

        // If there's an init declaration, execute it to populate fields.
        if let Some(init_decl) = init_decl {
            let env = Env::child(Rc::clone(captured));

            // Bind init params. Params are defined first.
            let mut positional: std::collections::VecDeque<Value> = args.into_iter().collect();
            let param_names: std::collections::HashSet<String> = init_decl.params.iter().map(|p| p.name.clone()).collect();
            for param in &init_decl.params {
                let val = if let Some(v) = positional.pop_front() { v } else { Value::Nil };
                env.borrow_mut().define_mut(&param.name, val);
            }

            // Pre-populate self fields as mutable locals — only those whose name doesn't
            // conflict with a param (params shadow fields in the init body).
            for (_, (name, val)) in decl.fields.iter().zip(fields.iter()) {
                if !param_names.contains(name) {
                    env.borrow_mut().define_mut(name, val.clone());
                }
            }

            // Execute init body.
            match self.exec_block(&init_decl.body, Rc::clone(&env)) {
                Ok(_) | Err(Signal::Return(_)) => {}
                Err(e) => return Err(e),
            }

            // Read back field values from env.
            for (name, val) in fields.iter_mut() {
                if let Some(new_val) = env.borrow().get(name) {
                    *val = new_val;
                }
            }
        } else if !args.is_empty() {
            // No init: positional args map to fields in order.
            for (i, arg) in args.into_iter().enumerate() {
                if let Some((_, v)) = fields.get_mut(i) {
                    *v = arg;
                }
            }
        }

        Ok(make_object(decl.name.clone(), fields))
    }

    /// Execute a `kernel:` block body.
    ///
    /// Differs from a plain block in one respect: `Stmt::Expr` whose value is a
    /// `KernelHandle` is silently waited — the updated kernel object is written back
    /// to the variable the expression reads from (so `k(block=N)` updates `k` in
    /// place, matching the design-doc semantics of `var k =`).
    pub(crate) fn exec_kernel_block(&mut self, stmts: &[crate::ast::Stmt], env: EnvRef) -> Result<(), Signal> {
        let prev_kernel_ctx = self.kernel_context;
        self.kernel_context = true;

        let result = self.exec_kernel_block_inner(stmts, env);

        self.kernel_context = prev_kernel_ctx;
        result
    }

    fn exec_kernel_block_inner(&mut self, stmts: &[crate::ast::Stmt], env: EnvRef) -> Result<(), Signal> {
        use crate::ast::Stmt;
        use crate::ast::ExprKind;

        for stmt in stmts {
            match stmt {
                Stmt::Expr(e) => {
                    let val = self.eval_expr(e, Rc::clone(&env))?;
                    // A KernelHandle from a bare `k(block=N)` call — wait immediately and
                    // write the updated kernel back to the source variable.
                    if let Value::KernelHandle { result } = val {
                        let var_name = match &e.kind {
                            ExprKind::Call(callee, _) => {
                                if let ExprKind::Var(name) = &callee.kind { Some(name.clone()) } else { None }
                            }
                            _ => None,
                        };
                        if let Some(name) = var_name {
                            env.borrow_mut().force_set(&name, *result);
                        }
                    }
                }
                Stmt::Loop(l) => {
                    // Inner `loop:` — GPU-driven render loop (simulation: run until `break`).
                    match self.exec_loop(l, Rc::clone(&env)) {
                        Ok(_) | Err(Signal::Break(_)) => {}
                        Err(e) => return Err(e),
                    }
                }
                other => {
                    self.exec_stmt(other, Rc::clone(&env))?;
                }
            }
        }
        Ok(())
    }
}

// Re-export FieldBinding for use in this file.
use crate::ast::{FieldBinding, Expr};
