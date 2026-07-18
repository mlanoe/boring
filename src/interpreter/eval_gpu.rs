// GPU kernel interpreter — simulation mode.
//
// In simulation mode (`boring run` or default), GPU kernels execute on the CPU.
// Each kernel thread runs independently with its own interpreter instance,
// allowing rayon to parallelize across threads while the main interpreter
// stays single-threaded (Rc-based).
//
// Parallelism strategy:
//   1. Snapshot kernel field values and the captured env into `ThreadValue`
//      (a Send+Sync mirror of `Value` that contains no Rc).
//   2. Use rayon to run each thread's entry-point body concurrently,
//      each with a fresh `Interpreter::new_for_kernel`.
//   3. Merge per-thread results back: scalars take the last write;
//      arrays do element-wise merge (GPU contract: non-overlapping writes).
//   4. Wrap the merged kernel object in a `KernelHandle`.

use super::*;
use std::rc::Rc;
use std::sync::OnceLock;
use rayon::prelude::*;

// Thread pool with an enlarged stack (64 MB) for the recursive tree-walk interpreter.
static KERNEL_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

fn kernel_pool() -> &'static rayon::ThreadPool {
    KERNEL_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .stack_size(64 * 1024 * 1024)
            .build()
            .expect("failed to build kernel thread pool")
    })
}

// ─── ThreadValue ─────────────────────────────────────────────────────────────
// A Send+Sync version of Value for passing data across kernel threads.
// Covers every type that can appear as a kernel field or env binding.
// Closures and channels are omitted (not valid kernel data).

#[derive(Clone)]
pub(crate) enum ThreadValue {
    Nil,
    Void,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    Str(String),
    Array(Vec<ThreadValue>),
    Object { type_name: String, fields: Vec<(String, ThreadValue)> },
    EnumVariant { type_name: String, variant: String, fields: Vec<ThreadValue> },
    Fn(Box<crate::ast::FnDecl>),
}

impl PartialEq for ThreadValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ThreadValue::Nil,    ThreadValue::Nil)    => true,
            (ThreadValue::Void,   ThreadValue::Void)   => true,
            (ThreadValue::Bool(a),  ThreadValue::Bool(b))  => a == b,
            (ThreadValue::Int(a),   ThreadValue::Int(b))   => a == b,
            (ThreadValue::Uint(a),  ThreadValue::Uint(b))  => a == b,
            (ThreadValue::Float(a), ThreadValue::Float(b)) => a == b,
            (ThreadValue::Str(a),   ThreadValue::Str(b))   => a == b,
            (ThreadValue::Array(a), ThreadValue::Array(b)) => a == b,
            (ThreadValue::Object { type_name: tn1, fields: f1 },
             ThreadValue::Object { type_name: tn2, fields: f2 }) => tn1 == tn2 && f1 == f2,
            (ThreadValue::EnumVariant { type_name: t1, variant: v1, fields: f1 },
             ThreadValue::EnumVariant { type_name: t2, variant: v2, fields: f2 }) =>
                t1 == t2 && v1 == v2 && f1 == f2,
            // Fn values are never kernel field data — treat as unequal so writes are detected.
            (ThreadValue::Fn(_), ThreadValue::Fn(_)) => false,
            _ => false,
        }
    }
}

fn to_thread_value(v: &Value) -> Option<ThreadValue> {
    match v {
        Value::Nil                => Some(ThreadValue::Nil),
        Value::Void               => Some(ThreadValue::Void),
        Value::Bool(b)            => Some(ThreadValue::Bool(*b)),
        Value::Int(n)             => Some(ThreadValue::Int(*n)),
        Value::Uint(n)            => Some(ThreadValue::Uint(*n)),
        Value::Float(f)           => Some(ThreadValue::Float(*f)),
        Value::Str(s)             => Some(ThreadValue::Str(s.clone())),
        Value::Fn { decl, .. }    => Some(ThreadValue::Fn(Box::new(decl.clone()))),
        Value::Array(arr)         => {
            let tvs: Option<Vec<_>> = arr.iter().map(to_thread_value).collect();
            tvs.map(ThreadValue::Array)
        }
        Value::Object(inner)      => {
            let inner = inner.borrow();
            let fields: Option<Vec<_>> = inner.fields.iter()
                .map(|(k, v)| to_thread_value(v).map(|tv| (k.clone(), tv)))
                .collect();
            fields.map(|f| ThreadValue::Object { type_name: inner.type_name.clone(), fields: f })
        }
        Value::EnumVariant { type_name, variant, fields } => {
            let tvs: Option<Vec<_>> = fields.iter().map(to_thread_value).collect();
            tvs.map(|f| ThreadValue::EnumVariant {
                type_name: type_name.clone(),
                variant: variant.clone(),
                fields: f,
            })
        }
        _ => None,
    }
}

fn from_thread_value(v: ThreadValue, captured: &EnvRef) -> Value {
    match v {
        ThreadValue::Nil              => Value::Nil,
        ThreadValue::Void             => Value::Void,
        ThreadValue::Bool(b)          => Value::Bool(b),
        ThreadValue::Int(n)           => Value::Int(n),
        ThreadValue::Uint(n)          => Value::Uint(n),
        ThreadValue::Float(f)         => Value::Float(f),
        ThreadValue::Str(s)           => Value::Str(s),
        ThreadValue::Fn(decl)         => Value::Fn { decl: *decl, captured: Rc::clone(captured) },
        ThreadValue::Array(arr)       => Value::Array(
            arr.into_iter().map(|tv| from_thread_value(tv, captured)).collect::<Vec<_>>().into()
        ),
        ThreadValue::Object { type_name, fields } => {
            let fields = fields.into_iter()
                .map(|(k, tv)| (k, from_thread_value(tv, captured)))
                .collect();
            Value::Object(Rc::new(RefCell::new(ObjectInner { type_name, fields })))
        }
        ThreadValue::EnumVariant { type_name, variant, fields } => Value::EnumVariant {
            type_name,
            variant,
            fields: fields.into_iter().map(|tv| from_thread_value(tv, captured)).collect(),
        },
    }
}

// ─── Snapshot helpers ─────────────────────────────────────────────────────────

/// Walk the env chain and snapshot every binding that converts to ThreadValue.
/// Stdlib functions are skipped (NativeFn is not convertible; the fresh interpreter
/// already provides them).
fn snapshot_env(env: &EnvRef) -> Vec<(String, ThreadValue)> {
    let mut out = Vec::new();
    let borrowed = env.borrow();
    for (name, val) in &borrowed.vars {
        if let Some(tv) = to_thread_value(val) {
            out.push((name.clone(), tv));
        }
    }
    if let Some(parent) = &borrowed.parent {
        // Walk parent chain but don't re-export stdlib (parent of global is None).
        out.extend(snapshot_env(parent));
    }
    out
}

// ─── Per-thread result ────────────────────────────────────────────────────────

struct ThreadResult {
    /// Changed mutable fields: (field_name, initial_value, new_value).
    /// Stored as ThreadValue so it's Send.
    fields: Vec<(String, ThreadValue, ThreadValue)>,
}

// ─── Core kernel simulation ───────────────────────────────────────────────────

/// Thread/block/grid dimensions for a single kernel launch (`run_kernel_parallel`).
pub(crate) struct LaunchDims {
    pub total_threads: usize,
    pub block_x:       usize,
    pub block_y:       usize,
    pub grid_x:        usize,
    pub grid_y:        usize,
}

/// Run the kernel's anonymous entry point for `dims.total_threads` threads in parallel.
///
/// Returns the final kernel object with all thread writes merged.
fn run_kernel_parallel(
    interp:     &Interpreter,
    decl:       &crate::ast::KernelDecl,
    captured:   &EnvRef,
    kernel_obj: Value,
    entry:      &crate::ast::FnDecl,
    dims:       LaunchDims,
) -> Result<Value, Signal> {
    let LaunchDims { total_threads, block_x, block_y, grid_x, grid_y } = dims;
    // Snapshot the initial field values and the captured env once (before threads).
    let initial_fields: Vec<(String, ThreadValue)> = if let Value::Object(ref obj) = kernel_obj {
        obj.borrow().fields.iter()
            .map(|(name, val)| (name.clone(), to_thread_value(val).unwrap_or(ThreadValue::Nil)))
            .collect()
    } else {
        vec![]
    };

    let mutable_names: Vec<String> = decl.fields.iter()
        .filter(|f| matches!(f.binding, FieldBinding::Mut | FieldBinding::Var))
        .map(|f| f.name.clone())
        .collect();

    let captured_snapshot = snapshot_env(captured);
    let entry_body        = entry.body.clone();
    let decl_fields       = decl.fields.clone();
    let decl_methods      = decl.methods.clone();
    let traits            = interp.traits.clone();
    let enums_map         = interp.enums.clone();
    let aliases           = interp.aliases.clone();
    let gpu_profile       = interp.gpu_profile.clone();

    // Run all threads in parallel with a large-stack pool to accommodate the
    // recursive tree-walk interpreter.
    let thread_results: Vec<Result<ThreadResult, String>> = kernel_pool().install(|| {
        (0..total_threads).into_par_iter().map(|thread_idx| {
            let threads_per_block = block_x * block_y;
            let flat_block        = thread_idx / threads_per_block;
            let flat_thread       = thread_idx % threads_per_block;
            let block_idx_x       = flat_block % grid_x;
            let block_idx_y       = flat_block / grid_x;
            let thread_in_x       = flat_thread % block_x;
            let thread_in_y       = flat_thread / block_x;

            // Build a fresh interpreter for this thread.
            let mut ti = Interpreter::new_for_kernel(
                traits.clone(),
                enums_map.clone(),
                aliases.clone(),
                gpu_profile.clone(),
            );

            // Reconstruct the captured env as a child of the fresh global.
            let cap_env = Env::child(Rc::clone(&ti.global));
            for (name, tv) in &captured_snapshot {
                let val = from_thread_value(tv.clone(), &cap_env);
                cap_env.borrow_mut().define(name, val);
            }

            // Build the thread env.
            let thread_env = Env::child(Rc::clone(&cap_env));

            // Inject field values (mutable fields are define_mut so the body can write them).
            for (field_decl, (name, tv)) in decl_fields.iter().zip(initial_fields.iter()) {
                let val = from_thread_value(tv.clone(), &thread_env);
                match field_decl.binding {
                    FieldBinding::Mut | FieldBinding::Var =>
                        thread_env.borrow_mut().define_mut(name, val),
                    FieldBinding::Let =>
                        thread_env.borrow_mut().define(name, val),
                }
            }

            // Reconstruct the kernel object as `self`.
            let self_fields: Vec<(String, Value)> = initial_fields.iter()
                .map(|(n, tv)| (n.clone(), from_thread_value(tv.clone(), &thread_env)))
                .collect();
            let self_obj = Value::Object(Rc::new(RefCell::new(ObjectInner {
                type_name: decl_fields.first().map(|_| "".to_string()).unwrap_or_default(),
                fields: self_fields,
            })));
            thread_env.borrow_mut().define("self", self_obj);

            // Inject gpu.* builtins.
            let gpu_thread = make_object("GpuThread".into(), vec![
                ("x".into(), Value::Int(thread_in_x as i64)),
                ("y".into(), Value::Int(thread_in_y as i64)),
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
            thread_env.borrow_mut().define("gpu", make_object("Gpu".into(), vec![
                ("thread".into(),    gpu_thread),
                ("block".into(),     gpu_block),
                ("block_dim".into(), gpu_block_dim),
                ("grid_dim".into(),  gpu_grid_dim),
            ]));

            // Inject kernel methods.
            for method in &decl_methods {
                if !method.name.is_empty() {
                    let fn_val = Value::Fn { decl: method.clone(), captured: Rc::clone(&thread_env) };
                    thread_env.borrow_mut().define(&method.name, fn_val);
                }
            }

            // Run the entry point.
            let result = ti.exec_block(&entry_body, Rc::clone(&thread_env));
            match result {
                Ok(_) | Err(Signal::Return(_)) => {}
                Err(Signal::Error(e)) => return Err(e.message),
                Err(e) => return Err(format!("{:?}", e)),
            }

            // Collect changed mutable fields.
            let mut changed = Vec::new();
            for name in &mutable_names {
                let initial = initial_fields.iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, tv)| tv.clone())
                    .unwrap_or(ThreadValue::Nil);
                if let Some(new_val) = thread_env.borrow().vars.get(name).and_then(to_thread_value) {
                    if new_val != initial {
                        changed.push((name.clone(), initial, new_val));
                    }
                }
            }
            Ok(ThreadResult { fields: changed })
        })
        .collect::<Vec<_>>()
    }); // kernel_pool().install

    // Check for any thread errors.
    let mut all_results = Vec::with_capacity(total_threads);
    for r in thread_results {
        match r {
            Ok(tr) => all_results.push(tr),
            Err(msg) => return Err(err(msg, 0)),
        }
    }

    // Merge results back into kernel_obj using element-wise merge for arrays.
    if let Value::Object(ref obj) = kernel_obj {
        for name in &mutable_names {
            let base_tv = initial_fields.iter()
                .find(|(n, _)| n == name)
                .map(|(_, tv)| tv.clone())
                .unwrap_or(ThreadValue::Nil);

            let merged_tv = match &base_tv {
                ThreadValue::Array(base_arr) => {
                    // Element-wise merge: for each element, take the first thread write that
                    // differs from the base value. GPU contract guarantees non-overlapping
                    // writes, so at most one thread changes each index.
                    let mut merged = base_arr.clone();
                    for tr in &all_results {
                        for (f_name, f_init, f_new) in &tr.fields {
                            if f_name != name { continue; }
                            if let (ThreadValue::Array(init_arr), ThreadValue::Array(new_arr)) =
                                (f_init, f_new)
                            {
                                for (i, (init_elem, new_elem)) in
                                    init_arr.iter().zip(new_arr.iter()).enumerate()
                                {
                                    if new_elem != init_elem {
                                        if let Some(slot) = merged.get_mut(i) {
                                            *slot = new_elem.clone();
                                        }
                                    }
                                }
                            }
                        }
                    }
                    ThreadValue::Array(merged)
                }
                _ => {
                    // Scalar: last write wins (matches sequential semantics).
                    let mut last = base_tv;
                    for tr in &all_results {
                        for (f_name, _, f_new) in &tr.fields {
                            if f_name == name {
                                last = f_new.clone();
                            }
                        }
                    }
                    last
                }
            };

            // Write merged value back into the shared kernel object.
            let val = from_thread_value(merged_tv, &Rc::new(RefCell::new(Env {
                parent: None,
                vars: Default::default(),
                mutable: Default::default(),
                owned_collections: Default::default(),
                task_safe_vars: Default::default(),
                owned_vars: Default::default(),
                shared_bindings: Default::default(),
                actor_bindings: Default::default(),
                lazy_vars: Default::default(),
            })));
            let mut o = obj.borrow_mut();
            if let Some(entry) = o.fields.iter_mut().find(|(n, _)| n == name) {
                entry.1 = val;
            }
        }
    }

    Ok(kernel_obj)
}

// ─── Interpreter impl ─────────────────────────────────────────────────────────

impl Interpreter {
    /// Register a `kernel Name:` declaration in the environment.
    pub(crate) fn exec_kernel_decl(&mut self, decl: &crate::ast::KernelDecl, env: EnvRef) -> Result<(), Signal> {
        let val = Value::KernelStruct { decl: decl.clone(), captured: Rc::clone(&env) };
        env.borrow_mut().define(&decl.name, val.clone());
        if !Rc::ptr_eq(&env, &self.global) {
            self.global.borrow_mut().define(&decl.name, val);
        }
        Ok(())
    }

    /// Evaluate a `kernel(block = N, ...) expr` expression.
    pub(crate) fn eval_kernel_launch(
        &mut self,
        config: &crate::ast::KernelConfig,
        kernel_expr: &Expr,
        env: EnvRef,
    ) -> Eval {
        let kernel_val = self.eval_expr(kernel_expr, Rc::clone(&env))?;

        let (type_name, fields, decl, captured) = match &kernel_val {
            Value::Object(inner) => {
                let type_name = inner.borrow().type_name.clone();
                let fields    = inner.borrow().fields.clone();
                let kval      = env.borrow().get(&type_name);
                match kval {
                    Some(Value::KernelStruct { decl, captured }) => (type_name, fields, decl, captured),
                    _ => return Ok(Value::KernelHandle { result: Box::new(kernel_val) }),
                }
            }
            _ => return Ok(Value::KernelHandle { result: Box::new(kernel_val) }),
        };

        let (block_x, block_y) = if let Some(block_expr) = &config.block {
            match self.eval_expr(block_expr, Rc::clone(&env))? {
                Value::Int(n) => (n.max(1) as usize, 1),
                Value::Tuple(t) => {
                    let x = match t.first() { Some(Value::Int(n)) => (*n).max(1) as usize, _ => 1 };
                    let y = match t.get(1)  { Some(Value::Int(n)) => (*n).max(1) as usize, _ => 1 };
                    (x, y)
                }
                _ => (1, 1),
            }
        } else {
            (1, 1)
        };

        let (grid_x, grid_y) = if let Some(grid_expr) = &config.grid {
            match self.eval_expr(grid_expr, Rc::clone(&env))? {
                Value::Int(n) => (n.max(1) as usize, 1),
                Value::Tuple(t) => {
                    let x = match t.first() { Some(Value::Int(n)) => (*n).max(1) as usize, _ => 1 };
                    let y = match t.get(1)  { Some(Value::Int(n)) => (*n).max(1) as usize, _ => 1 };
                    (x, y)
                }
                _ => (1, 1),
            }
        } else {
            let max_len = fields.iter()
                .filter_map(|(_, v)| if let Value::Array(a) = v { Some(a.len()) } else { None })
                .max()
                .unwrap_or(0);
            let inferred_x = if max_len > 0 { max_len.div_ceil(block_x * block_y) } else { 1 };
            (inferred_x, 1)
        };

        let total_threads = block_x * block_y * grid_x * grid_y;

        let entry = decl.methods.iter().find(|m| m.name.is_empty() && m.params.is_empty());
        if let Some(entry) = entry {
            let kernel_obj = make_object(type_name, fields);
            let kernel_obj = run_kernel_parallel(
                self, &decl, &captured, kernel_obj, entry,
                LaunchDims { total_threads, block_x, block_y, grid_x, grid_y },
            )?;
            Ok(Value::KernelHandle { result: Box::new(kernel_obj) })
        } else {
            Ok(Value::KernelHandle { result: Box::new(make_object(type_name, fields)) })
        }
    }

    /// `k(block = N)` short-hand dispatch.
    pub(crate) fn eval_kernel_launch_with_val(
        &mut self,
        _config: crate::ast::KernelConfig,
        kernel_val: Value,
        block_val: Value,
        _line: usize,
        env: &EnvRef,
    ) -> Eval {
        let (type_name, fields, decl, captured) = match &kernel_val {
            Value::Object(inner) => {
                let type_name = inner.borrow().type_name.clone();
                let fields    = inner.borrow().fields.clone();
                let kval      = env.borrow().get(&type_name);
                match kval {
                    Some(Value::KernelStruct { decl, captured }) => (type_name, fields, decl, captured),
                    _ => return Ok(Value::KernelHandle { result: Box::new(kernel_val) }),
                }
            }
            _ => return Ok(Value::KernelHandle { result: Box::new(kernel_val) }),
        };

        let (block_x, block_y) = match block_val {
            Value::Int(n)   => (n.max(1) as usize, 1),
            Value::Uint(n)  => (n.max(1) as usize, 1),
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

        let max_len = fields.iter()
            .filter_map(|(_, v)| if let Value::Array(a) = v { Some(a.len()) } else { None })
            .max()
            .unwrap_or(0);

        let (grid_x, grid_y) = if max_len > 0 {
            let bxy = block_x * block_y;
            let dim_w = fields.iter().find(|(n, _)| n == "dim")
                .and_then(|(_, v)| if let Value::Object(o) = v {
                    o.borrow().fields.iter().find(|(k, _)| k == "width")
                        .and_then(|(_, w)| if let Value::Uint(n) = w { Some(*n as usize) } else { None })
                } else { None });
            let dim_h = fields.iter().find(|(n, _)| n == "dim")
                .and_then(|(_, v)| if let Value::Object(o) = v {
                    o.borrow().fields.iter().find(|(k, _)| k == "height")
                        .and_then(|(_, h)| if let Value::Uint(n) = h { Some(*n as usize) } else { None })
                } else { None });
            if let (Some(w), Some(h)) = (dim_w, dim_h) {
                (w.div_ceil(block_x), h.div_ceil(block_y))
            } else {
                (max_len.div_ceil(bxy), 1)
            }
        } else {
            (1, 1)
        };

        let total_threads = block_x * block_y * grid_x * grid_y;

        let entry = decl.methods.iter().find(|m| m.name.is_empty() && m.params.is_empty());
        if let Some(entry) = entry {
            let kernel_obj = make_object(type_name, fields);
            let kernel_obj = run_kernel_parallel(
                self, &decl, &captured, kernel_obj, entry,
                LaunchDims { total_threads, block_x, block_y, grid_x, grid_y },
            )?;
            Ok(Value::KernelHandle { result: Box::new(kernel_obj) })
        } else {
            Ok(Value::KernelHandle { result: Box::new(make_object(type_name, fields)) })
        }
    }

    /// Instantiate a kernel struct: call its `init` declaration (if any).
    pub(crate) fn instantiate_kernel_struct(
        &mut self,
        decl: &crate::ast::KernelDecl,
        captured: &EnvRef,
        args: Vec<Value>,
        _line: usize,
    ) -> Eval {
        let mut fields: Vec<(String, Value)> = decl.fields
            .iter()
            .map(|f| (f.name.clone(), Value::Nil))
            .collect();

        let init_decl = decl.inits.iter()
            .find(|i| {
                if i.params.len() != args.len() { return false; }
                if let (Some(first_param), Some(first_arg)) = (i.params.first(), args.first()) {
                    let expected = first_param.ty.as_ref().map(|t| match t {
                        crate::ast::Type::Named(n) => n.as_str(),
                        _ => "",
                    });
                    let actual = first_arg.type_name();
                    if let Some(exp) = expected {
                        if !exp.is_empty() && exp != actual.as_str() { return false; }
                    }
                }
                true
            })
            .or_else(|| decl.inits.first());

        if let Some(init_decl) = init_decl {
            let env = Env::child(Rc::clone(captured));
            let mut positional: std::collections::VecDeque<Value> = args.into_iter().collect();
            let param_names: std::collections::HashSet<String> =
                init_decl.params.iter().map(|p| p.name.clone()).collect();
            for param in &init_decl.params {
                let val = positional.pop_front().unwrap_or(Value::Nil);
                env.borrow_mut().define_mut(&param.name, val);
            }
            for (_, (name, val)) in decl.fields.iter().zip(fields.iter()) {
                if !param_names.contains(name) {
                    env.borrow_mut().define_mut(name, val.clone());
                }
            }
            match self.exec_block(&init_decl.body, Rc::clone(&env)) {
                Ok(_) | Err(Signal::Return(_)) => {}
                Err(e) => return Err(e),
            }
            for (name, val) in fields.iter_mut() {
                if let Some(new_val) = env.borrow().get(name) {
                    *val = new_val;
                }
            }
        } else if !args.is_empty() {
            for (i, arg) in args.into_iter().enumerate() {
                if let Some((_, v)) = fields.get_mut(i) {
                    *v = arg;
                }
            }
        }

        Ok(make_object(decl.name.clone(), fields))
    }

    /// Execute a `kernel:` block body.
    pub(crate) fn exec_kernel_block(&mut self, stmts: &[crate::ast::Stmt], env: EnvRef) -> Result<(), Signal> {
        let prev = self.kernel_context;
        self.kernel_context = true;
        let result = self.exec_kernel_block_inner(stmts, env);
        self.kernel_context = prev;
        result
    }

    fn exec_kernel_block_inner(&mut self, stmts: &[crate::ast::Stmt], env: EnvRef) -> Result<(), Signal> {
        use crate::ast::Stmt;
        use crate::ast::ExprKind;

        for stmt in stmts {
            match stmt {
                Stmt::Expr(e) => {
                    let val = self.eval_expr(e, Rc::clone(&env))?;
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

use crate::ast::{FieldBinding, Expr};
