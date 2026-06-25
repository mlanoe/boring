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
            // Determine block size.
            let block_x = if let Some(block_expr) = &config.block {
                match self.eval_expr(block_expr, Rc::clone(&env))? {
                    Value::Int(n) => n.max(1) as usize,
                    Value::Tuple(t) => match t.first() {
                        Some(Value::Int(n)) => (*n).max(1) as usize,
                        _ => 1,
                    },
                    _ => 1,
                }
            } else {
                1
            };

            // Determine grid size (number of blocks).
            let grid_x = if let Some(grid_expr) = &config.grid {
                match self.eval_expr(grid_expr, Rc::clone(&env))? {
                    Value::Int(n) => n.max(1) as usize,
                    Value::Tuple(t) => match t.first() {
                        Some(Value::Int(n)) => (*n).max(1) as usize,
                        _ => 1,
                    },
                    _ => 1,
                }
            } else {
                1  // default: 1 block (no data-parallel grid inference in simulation)
            };

            let total_threads = block_x * grid_x;

            // Build the kernel object to mutate during simulation.
            let kernel_obj = make_object(type_name.clone(), fields);

            // Run the entry point for each thread sequentially.
            for thread_idx in 0..total_threads {
                let block_idx = thread_idx / block_x;
                let thread_in_block = thread_idx % block_x;

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
                    ("x".into(), Value::Int(thread_in_block as i64)),
                    ("y".into(), Value::Int(0)),
                    ("z".into(), Value::Int(0)),
                ]);
                let gpu_block = make_object("GpuBlock".into(), vec![
                    ("x".into(), Value::Int(block_idx as i64)),
                    ("y".into(), Value::Int(0)),
                    ("z".into(), Value::Int(0)),
                ]);
                let gpu_block_dim = make_object("GpuBlockDim".into(), vec![
                    ("x".into(), Value::Int(block_x as i64)),
                    ("y".into(), Value::Int(1)),
                    ("z".into(), Value::Int(1)),
                ]);
                let gpu_grid_dim = make_object("GpuGridDim".into(), vec![
                    ("x".into(), Value::Int(grid_x as i64)),
                    ("y".into(), Value::Int(1)),
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

        // If there's an init declaration, execute it to populate fields.
        if let Some(init_decl) = decl.inits.first() {
            let env = Env::child(Rc::clone(captured));

            // Bind init params.
            let mut positional: std::collections::VecDeque<Value> = args.into_iter().collect();
            for param in &init_decl.params {
                let val = if let Some(v) = positional.pop_front() { v } else { Value::Nil };
                env.borrow_mut().define(&param.name, val);
            }

            // Pre-populate self fields as mutable locals in the init body.
            // All fields are writable during init (first assignment from constructor).
            for (_, (name, val)) in decl.fields.iter().zip(fields.iter()) {
                env.borrow_mut().define_mut(name, val.clone());
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
}

// Re-export FieldBinding for use in this file.
use crate::ast::{FieldBinding, Expr};
