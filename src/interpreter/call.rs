use super::*;
use std::collections::HashMap;
use std::rc::Rc;

impl Interpreter {
    pub(crate) fn call_fn(&mut self, decl: &FnDecl, captured: EnvRef, args: Vec<Value>, line: usize, _in_throws_context: bool) -> Eval {
        // Stream functions: run the body collecting `yield` values into a Vec.
        if decl.stream {
            return self.call_stream_fn(decl, captured, args, line);
        }
        let fn_env = Env::child(captured.clone());

        // Bind params — supports labeled args, default values, and variadic params.
        // `args` is a flat Vec<Value> whose items may carry an optional label
        // (encoded as Value::Labeled { label, value } — see eval_args_labeled).
        // Algorithm:
        //   1. Separate labeled from positional args.
        //   2. For each param in order:
        //      a. If it's variadic, collect all remaining positional args as an Array.
        //      b. Otherwise try labeled first, then positional, then default, then Nil.

        // Split args into labeled and positional pools
        let mut labeled_pool: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
        let mut positional_pool: std::collections::VecDeque<Value> = std::collections::VecDeque::new();
        for val in args {
            match val {
                Value::Labeled { label, value } => { labeled_pool.insert(label, *value); }
                other => positional_pool.push_back(other),
            }
        }

        for param in &decl.params {
            let val = if param.variadic {
                // Collect all remaining positional values into an Array
                let remaining: Vec<Value> = positional_pool.drain(..).collect();
                Value::Array(remaining.into())
            } else if let Some(v) = labeled_pool.remove(&param.name) {
                v
            } else if let Some(v) = positional_pool.pop_front() {
                v
            } else if let Some(default_expr) = &param.default {
                // Evaluate default in the captured (caller) environment
                self.eval_expr(default_expr, Rc::clone(&captured))?
            } else {
                Value::Nil
            };

            if param.mutable {
                fn_env.borrow_mut().define_mut(&param.name, val);
                // Parameter `mut`/`var` still both grant full access (rebind +
                // content mutation) — the parameter model's own three/four-way
                // split (docs/mut-type-modifier.md's "Parameters" section) is
                // specified but not enforced yet, deliberately out of scope
                // here; this just keeps that existing, unchanged behavior.
                fn_env.borrow_mut().mark_content_mutable(&param.name);
            } else {
                fn_env.borrow_mut().define(&param.name, val);
            }
        }

        // ── Generic: infer type-parameter bindings and check where clause ──────────
        let has_type_params = !decl.type_params.is_empty();
        if has_type_params {
            let mut bindings = HashMap::new();
            for param in &decl.params {
                if let Some(ty) = &param.ty {
                    let val = fn_env.borrow().get(&param.name).unwrap_or(Value::Nil);
                    let actual_ty = Self::type_of_value(&val);
                    Self::infer_from_type_params(&decl.type_params, ty, &actual_ty, &mut bindings);
                }
            }
            self.check_where_clause(&decl.where_clause, &bindings, line)?;
            self.type_param_stack.push(bindings);
        } else if !decl.where_clause.is_empty() {
            // Non-generic function with an explicit where clause: bindings are empty,
            // so constraints are trivially unsatisfiable — report as an error.
            return Err(err(
                format!("function '{}' has a where clause but no type parameters", decl.name),
                line,
            ));
        }

        // ── Param type checking (after type-param stack is populated) ────────────
        let params_clone = decl.params.clone();
        for param in &params_clone {
            let Some(ref ann_ty) = param.ty else { continue };
            let resolved_ty = self.resolve_type(ann_ty);
            let val = fn_env.borrow().get(&param.name).unwrap_or(Value::Nil);

            if param.variadic {
                // Variadic: val is an Array; check each element against the declared element type.
                if let Value::Array(ref elems) = val {
                    for (i, elem) in elems.iter().enumerate() {
                        let coerced = Self::coerce_to_type(elem.clone(), &resolved_ty);
                        if !self.value_matches_type(&coerced, &resolved_ty) {
                            if has_type_params { self.type_param_stack.pop(); }
                            return Err(err(
                                format!(
                                    "variadic argument '{}[{}]': expected {}, got {}",
                                    param.name, i, Self::display_type(&resolved_ty), coerced.type_name()
                                ),
                                line,
                            ));
                        }
                    }
                }
                continue;
            }

            let coerced = Self::coerce_to_type(val, &resolved_ty);
            if !self.value_matches_type(&coerced, &resolved_ty) {
                // Try implicit user-defined `as T:` conversion before erroring.
                match self.cast_value(coerced.clone(), &resolved_ty, line) {
                    Ok(converted) if self.value_matches_type(&converted, &resolved_ty) => {
                        fn_env.borrow_mut().force_set(&param.name, converted);
                        continue;
                    }
                    _ => {
                        if has_type_params { self.type_param_stack.pop(); }
                        return Err(err(
                            format!(
                                "argument '{}': expected {}, got {}",
                                param.name, Self::display_type(&resolved_ty), coerced.type_name()
                            ),
                            line,
                        ));
                    }
                }
            }
            // Write back the coerced value (e.g. Int→Uint).
            fn_env.borrow_mut().force_set(&param.name, coerced);
        }

        // ── Resolve return type now, while the type-param stack is still live ────
        let resolved_ret = decl.return_ty.as_ref().map(|t| self.resolve_type(t));

        let prev_task_ctx = self.task_context;
        // `main` is always treated as a task context — there is no caller that
        // needs to know, so `def main():` works the same as `task main():`.
        self.task_context = decl.task || decl.name == "main";
        let prev_mutating = self.current_method_mutating;
        self.current_method_mutating = decl.mutating;
        // Push a defer frame for this call
        self.defer_stack.push(Vec::new());
        // Execute every statement EXCEPT the last (defers are only registered here,
        // not yet executed).  Any early Return / error propagates immediately.
        let pre_result = self.exec_all_but_last(&decl.body, Rc::clone(&fn_env));
        // NOTE: task_context and current_method_mutating are restored AFTER the
        // tail expression and deferred blocks execute — they must remain set to
        // the callee's values for the entire duration of the call.

        // Run deferred blocks in LIFO order.
        let run_defers = |interp: &mut Self, env: EnvRef| {
            if let Some(frame) = interp.defer_stack.pop() {
                for deferred in frame.into_iter().rev() {
                    let _ = interp.exec_block(&deferred, Rc::clone(&env));
                }
            }
        };

        // Determine whether the last statement produces a return value.
        // - A bare expression (variable read, arithmetic, if/match expr) → value-producing
        // - An assignment expression, `defer`, or any other statement → non-value-producing
        let last_produces_value = decl.body.last().map(|s| match s {
            Stmt::Expr(e) => !matches!(e.kind, ExprKind::Assign(..)),
            Stmt::If(_) | Stmt::Match(_) => true,
            _ => false,
        }).unwrap_or(false);

        // Evaluate the tail expression.
        let result: Eval = match pre_result {
            Ok(()) => {
                if last_produces_value {
                    // Value-producing last expression: run defers FIRST so that any
                    // mutations they make to local variables are visible in the return value.
                    // e.g. `var log = ""; defer: log += "+closed"; log += "x"; log`
                    //       → defers run → log == "x+closed" → return "x+closed"
                    run_defers(self, Rc::clone(&fn_env));
                    if let Some(last) = decl.body.last() {
                        match self.eval_tail_stmt(last, Rc::clone(&fn_env)) {
                            Err(Signal::Return(v)) => Ok(v),
                            other => other,
                        }
                    } else {
                        Ok(Value::Nil)
                    }
                } else {
                    // Non-value last statement (assignment, defer, void call…):
                    // execute the last statement first, THEN run defers.
                    // e.g. `defer: log += "d"; log += "body "` → "body d"
                    let tail_result = if let Some(last) = decl.body.last() {
                        self.exec_stmt(last, Rc::clone(&fn_env))
                    } else {
                        Ok(())
                    };
                    run_defers(self, Rc::clone(&fn_env));
                    match tail_result {
                        Ok(()) => Ok(Value::Nil),
                        Err(Signal::Return(v)) => Ok(v),
                        Err(other) => Err(other),
                    }
                }
            }
            Err(Signal::Return(v)) => {
                run_defers(self, Rc::clone(&fn_env));
                Ok(v)
            }
            Err(other) => {
                run_defers(self, Rc::clone(&fn_env));
                Err(other)
            }
        };

        // Restore caller's context flags now that the tail expression and all
        // deferred blocks have finished executing.
        self.task_context = prev_task_ctx;
        self.current_method_mutating = prev_mutating;

        // Pop type-parameter bindings
        if has_type_params {
            self.type_param_stack.pop();
        }

        // Populate last_var_params so the call site can write back mutated var args.
        self.last_var_params.clear();
        for param in &decl.params {
            if param.mutable {
                if let Some(val) = fn_env.borrow().get(&param.name) {
                    self.last_var_params.insert(param.name.clone(), val);
                }
            }
        }

        match result {
            Ok(v) => {
                if resolved_ret.as_ref().map(|t| matches!(t, Type::Void)).unwrap_or(false) {
                    Ok(Value::Void)
                } else if let Some(ref resolved) = resolved_ret {
                    let coerced = Self::coerce_to_type(v, resolved);
                    // ── Return type check ────────────────────────────────────────
                    if !self.value_matches_type(&coerced, resolved) {
                        return Err(err(
                            format!(
                                "function '{}' declared to return {}, but got {}",
                                decl.name, Self::display_type(resolved), coerced.type_name()
                            ),
                            line,
                        ));
                    }
                    Ok(coerced)
                } else {
                    Ok(v)
                }
            }
            Err(Signal::Exception(v)) => {
                if decl.throws {
                    Err(Signal::Exception(v))
                } else {
                    Err(err(
                        format!("unhandled exception in non-throws function '{}': {}", decl.name, v),
                        line,
                    ))
                }
            }
            Err(other) => Err(other),
        }
    }

    /// Execute a stream function, collecting all `yield`ed values into an Array.
    /// Yields are captured as side effects into `self.stream_yields` (no signal propagation),
    /// so `for` loops and other control flow inside the stream body work normally.
    pub(crate) fn call_stream_fn(&mut self, decl: &FnDecl, captured: EnvRef, args: Vec<Value>, _line: usize) -> Eval {
        let fn_env = Env::child(captured.clone());
        // Bind params positionally
        let mut pos_iter = args.into_iter();
        for param in &decl.params {
            let val = pos_iter.next().unwrap_or(Value::Nil);
            fn_env.borrow_mut().define(&param.name, val);
        }
        // Save + set stream context
        let prev_in_stream = self.in_stream;
        let prev_yields   = std::mem::take(&mut self.stream_yields);
        self.in_stream = true;

        let result = self.exec_block(&decl.body, Rc::clone(&fn_env));

        // Restore context and harvest collected values
        self.in_stream = prev_in_stream;
        let collected = std::mem::replace(&mut self.stream_yields, prev_yields);

        match result {
            Ok(()) | Err(Signal::Return(_)) => Ok(Value::Array(collected.into())),
            Err(other) => Err(other),
        }
    }

    pub(crate) fn call_closure(&mut self, params: Vec<Param>, body: ClosureBody, captured: EnvRef, args: Vec<Value>, _line: usize) -> Eval {
        let fn_env = Env::child(captured);
        for (i, param) in params.iter().enumerate() {
            let val = args.get(i).cloned().unwrap_or(Value::Nil);
            fn_env.borrow_mut().define(&param.name, val);
        }
        match body {
            ClosureBody::Expr(e) => self.eval_expr(&e, fn_env),
            ClosureBody::Block(stmts) => {
                // Absorb break/continue at the closure boundary so they don't
                // escape into the enclosing loop. `break val` returns `val`;
                // bare `break` and `continue` return nil.
                match self.eval_block_as_expr(&stmts, fn_env) {
                    Err(Signal::Break(val)) => Ok(val),
                    Err(Signal::Continue)   => Ok(Value::Nil),
                    other                   => other,
                }
            }
        }
    }

    /// Execute a `type def/req/set` method. No `self` — type_vars accessed via type_var_store.
    pub(crate) fn call_type_method(
        &mut self,
        type_name: &str,
        method: &crate::ast::TypeMethod,
        args: Vec<Value>,
        captured: EnvRef,
        _line: usize,
    ) -> Eval {
        self.defer_stack.push(Vec::new());
        let fn_env = Env::child(Rc::clone(&captured));
        // Bind parameters
        for (i, param) in method.params.iter().enumerate() {
            let val = args.get(i).cloned().unwrap_or(Value::Nil);
            fn_env.borrow_mut().define(&param.name, val);
        }
        // Make `TypeName` available in the body so `Counter.MAX` etc. resolve
        // (the global env is already a parent via the captured chain — no extra binding needed)
        let _ = type_name; // used by caller for context; body resolves via global chain
        // Execute all but the last statement, then run defers, then evaluate the tail.
        let pre = self.exec_all_but_last(&method.body, Rc::clone(&fn_env));
        let defers = self.defer_stack.pop().unwrap_or_default();
        for block in defers.into_iter().rev() {
            let _ = self.eval_block_as_expr(&block, Rc::clone(&captured));
        }
        match pre {
            Ok(()) => {
                if let Some(last) = method.body.last() {
                    match self.eval_tail_stmt(last, fn_env) {
                        Err(Signal::Return(v)) => Ok(v),
                        other => other,
                    }
                } else {
                    Ok(Value::Nil)
                }
            }
            Err(Signal::Return(v)) => Ok(v),
            Err(other) => Err(other),
        }
    }

    pub(crate) fn instantiate_struct_labeled(&mut self, decl: &StructDecl, captured: &EnvRef, args: Vec<Value>, line: usize) -> Eval {

        if !decl.inits.is_empty() {
            let arg_count = args.len();
            let init_decl = decl.inits.iter()
                .min_by_key(|init| {
                    let total = init.params.len();
                    let required = init.params.iter().filter(|p| p.default.is_none()).count();
                    if arg_count >= required && arg_count <= total {
                        0usize
                    } else {
                        ((total as i64) - (arg_count as i64)).unsigned_abs() as usize + 1
                    }
                })
                .unwrap()
                .clone();
            return self.call_init(decl, &init_decl, captured, args, line);
        }

        // No `init` → field-based labeled/positional construction.
        let mut labeled: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
        let mut positional: std::collections::VecDeque<Value> = std::collections::VecDeque::new();
        for val in args {
            match val {
                Value::Labeled { label, value } => { labeled.insert(label, *value); }
                other => positional.push_back(other),
            }
        }
        let mut fields: Vec<(String, Value)> = Vec::new();
        for field_decl in &decl.fields {
            let val = if let Some(v) = labeled.remove(&field_decl.name) {
                v
            } else if let Some(v) = positional.pop_front() {
                v
            } else if let Some(default_expr) = &field_decl.default {
                let default_expr = default_expr.clone();
                self.eval_expr(&default_expr, Rc::clone(captured))?
            } else {
                Value::Nil
            };
            fields.push((field_decl.name.clone(), val));
        }
        Ok(make_object(decl.name.clone(), fields))
    }

    /// Execute an `init` declaration and return the constructed object.
    pub(crate) fn call_init(&mut self, decl: &StructDecl, init_decl: &crate::ast::InitDecl, captured: &EnvRef, args: Vec<Value>, _line: usize) -> Eval {
        // Resolve positional / labeled args.
        let mut positional: std::collections::VecDeque<Value> = std::collections::VecDeque::new();
        let mut labeled: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
        for val in args {
            match val {
                Value::Labeled { label, value } => { labeled.insert(label, *value); }
                other => positional.push_back(other),
            }
        }

        let _resolve_arg = |labeled: &mut std::collections::HashMap<String, Value>,
                           positional: &mut std::collections::VecDeque<Value>,
                           name: &str| -> Value {
            if let Some(v) = labeled.remove(name) { v }
            else if let Some(v) = positional.pop_front() { v }
            else { Value::Nil }
        };

        // ── No body: every param declares a struct field ──────────────────────
        if init_decl.body.is_empty() {
            let mut fields: Vec<(String, Value)> = Vec::new();
            let mut param_names: std::collections::HashSet<String> = std::collections::HashSet::new();

            for param in &init_decl.params {
                let val = if let Some(v) = labeled.remove(&param.name) {
                    v
                } else if let Some(v) = positional.pop_front() {
                    v
                } else if let Some(default_expr) = &param.default {
                    let default_expr = default_expr.clone();
                    self.eval_expr(&default_expr, Rc::clone(captured))?
                } else {
                    Value::Nil
                };
                param_names.insert(param.name.clone());
                fields.push((param.name.clone(), val));
            }

            // Append any struct-body fields not covered by init params (use their defaults).
            for field_decl in &decl.fields {
                if !param_names.contains(&field_decl.name) {
                    let val = if let Some(default_expr) = &field_decl.default {
                        let default_expr = default_expr.clone();
                        self.eval_expr(&default_expr, Rc::clone(captured))?
                    } else {
                        Value::Nil
                    };
                    fields.push((field_decl.name.clone(), val));
                }
            }

            return Ok(make_object(decl.name.clone(), fields));
        }

        // ── With body: params are plain locals, `self` pre-populated from struct fields ──
        let env = Env::child(Rc::clone(captured));

        // Bind params as locals.
        for param in &init_decl.params {
            let val = if let Some(v) = labeled.remove(&param.name) {
                v
            } else if let Some(v) = positional.pop_front() {
                v
            } else if let Some(default_expr) = &param.default {
                let default_expr = default_expr.clone();
                self.eval_expr(&default_expr, Rc::clone(captured))?
            } else {
                Value::Nil
            };
            if param.mutable {
                env.borrow_mut().define_mut(&param.name, val);
                // See the matching comment at this function's other call site.
                env.borrow_mut().mark_content_mutable(&param.name);
            } else {
                env.borrow_mut().define(&param.name, val);
            }
        }

        // Pre-populate `self` with field defaults from the struct body.
        let mut init_fields: Vec<(String, Value)> = Vec::new();
        for field_decl in &decl.fields {
            let val = if let Some(default_expr) = &field_decl.default {
                let default_expr = default_expr.clone();
                self.eval_expr(&default_expr, Rc::clone(captured))?
            } else {
                Value::Nil
            };
            init_fields.push((field_decl.name.clone(), val));
        }
        let self_obj = make_object(decl.name.clone(), init_fields);
        env.borrow_mut().define_mut("self", self_obj);

        // Run the body with mutating + init-body context.
        let prev_mutating = self.current_method_mutating;
        let prev_in_init = self.in_init_body;
        self.current_method_mutating = true;
        self.in_init_body = true;

        let result = self.exec_block(&init_decl.body, Rc::clone(&env));

        self.current_method_mutating = prev_mutating;
        self.in_init_body = prev_in_init;

        match result {
            Ok(_) | Err(crate::interpreter::Signal::Return(_)) => {}
            Err(e) => return Err(e),
        }

        let final_self = env.borrow().get("self").ok_or_else(|| {
            err("init body did not assign 'self'", 0)
        })?;
        Ok(final_self)
    }

    pub(crate) fn has_method(&self, type_name: &str, method: &str) -> bool {
        if let Some(val) = self.global.borrow().get(type_name) {
            match &val {
                Value::Struct { decl, .. }
                    if decl.methods.iter().any(|m| m.name == method) => { return true; }
                Value::EnumNamespace { methods, .. }
                    if methods.iter().any(|m| m.name == method) => { return true; }
                _ => {}
            }
        }
        false
    }

    pub(crate) fn try_operator_method(&mut self, obj: &Value, method: &str, rhs: Value, line: usize) -> Result<Option<Value>, Signal> {
        if let Value::Object(inner_rc) = obj {
            let type_name = inner_rc.borrow().type_name.clone();
            if self.has_method(&type_name, method) {
                let mut out_self = None;
                let result = self.call_method(obj.clone(), method, vec![rhs], line, &mut out_self)?;
                return Ok(Some(result));
            }
        }
        Ok(None)
    }

    // ─── Macro call dispatch ─────────────────────────────────────────────────

    pub(crate) fn call_macro(&mut self, name: &str, args: Vec<Value>, line: usize) -> Eval {
        match name {
            // ── I/O macros ───────────────────────────────────────────────────
            "println" => {
                println!("{}", Self::macro_format(&args, line)?);
                Ok(Value::Void)
            }
            "print" => {
                print!("{}", Self::macro_format(&args, line)?);
                Ok(Value::Void)
            }
            "eprintln" => {
                eprintln!("{}", Self::macro_format(&args, line)?);
                Ok(Value::Void)
            }
            "eprint" => {
                eprint!("{}", Self::macro_format(&args, line)?);
                Ok(Value::Void)
            }

            // ── format! ──────────────────────────────────────────────────────
            "format" => {
                Ok(Value::Str(Self::macro_format(&args, line)?))
            }

            // ── Collection constructors ───────────────────────────────────────
            // `vec![a, b, c]`  →  boring Array
            "vec" => Ok(Value::Array(args.into())),
            // `hashmap!{k => v, ...}` or `hashmap!(k, v, ...)` — pairs → Dict
            "hashmap" | "btreemap" => {
                let mut pairs = Vec::new();
                let mut i = 0;
                while i + 1 < args.len() {
                    pairs.push((args[i].clone(), args[i + 1].clone()));
                    i += 2;
                }
                Ok(Value::Dict(pairs))
            }

            // ── Assertion macros ─────────────────────────────────────────────
            "assert" => {
                match args.first() {
                    Some(Value::Bool(true)) => Ok(Value::Void),
                    Some(Value::Bool(false)) => {
                        let msg = args.get(1)
                            .map(|v| format!("{}", v))
                            .unwrap_or_else(|| "assertion failed".to_string());
                        Err(err(msg, line))
                    }
                    _ => Err(err("assert!: expected bool argument", line)),
                }
            }
            "assert_eq" => {
                let ok = match (args.first(), args.get(1)) {
                    (Some(a), Some(b)) => a == b,
                    _ => false,
                };
                if ok { Ok(Value::Void) }
                else {
                    let msg = args.get(2)
                        .map(|v| format!("{}", v))
                        .unwrap_or_else(|| format!("assertion `left == right` failed\n  left: {}\n right: {}",
                            args.first().unwrap_or(&Value::Nil),
                            args.get(1).unwrap_or(&Value::Nil)));
                    Err(err(msg, line))
                }
            }
            "assert_ne" => {
                let neq = match (args.first(), args.get(1)) {
                    (Some(a), Some(b)) => a != b,
                    _ => true,
                };
                if neq { Ok(Value::Void) }
                else {
                    let msg = args.get(2)
                        .map(|v| format!("{}", v))
                        .unwrap_or_else(|| format!("assertion `left != right` failed\n  value: {}",
                            args.first().unwrap_or(&Value::Nil)));
                    Err(err(msg, line))
                }
            }

            // ── Panic / control flow ─────────────────────────────────────────
            "panic" => {
                let msg = if args.is_empty() {
                    "explicit panic".to_string()
                } else {
                    Self::macro_format(&args, line)?
                };
                Err(err(msg, line))
            }
            "todo" => {
                let msg = if args.is_empty() {
                    "not yet implemented".to_string()
                } else {
                    Self::macro_format(&args, line)?
                };
                Err(err(format!("todo!: {}", msg), line))
            }
            "unimplemented" => {
                let msg = if args.is_empty() {
                    "not implemented".to_string()
                } else {
                    Self::macro_format(&args, line)?
                };
                Err(err(format!("unimplemented!: {}", msg), line))
            }
            "unreachable" => {
                let msg = if args.is_empty() {
                    "entered unreachable code".to_string()
                } else {
                    Self::macro_format(&args, line)?
                };
                Err(err(format!("unreachable!: {}", msg), line))
            }

            // ── Debug ────────────────────────────────────────────────────────
            "dbg" => {
                match args.into_iter().next() {
                    Some(v) => {
                        eprintln!("[dbg] {:?}", v);
                        Ok(v)
                    }
                    None => Ok(Value::Void),
                }
            }

            // ── write! / writeln! — best-effort: write to stdout ─────────────
            "write" => {
                // First arg is the writer (ignored in interpreter), rest is format.
                let rest: Vec<Value> = args.into_iter().skip(1).collect();
                print!("{}", Self::macro_format(&rest, line)?);
                Ok(Value::Void)
            }
            "writeln" => {
                let rest: Vec<Value> = args.into_iter().skip(1).collect();
                println!("{}", Self::macro_format(&rest, line)?);
                Ok(Value::Void)
            }

            // ── include_str! / env! — compile-time, return empty stub ────────
            "include_str" => Ok(Value::Str(String::new())),
            "env" => Ok(Value::Str(String::new())),
            "concat" => {
                let s: String = args.iter().map(|v| format!("{}", v)).collect();
                Ok(Value::Str(s))
            }

            // ── Unknown macro — produce Void (no crash on side-effect macros) ─
            _ => {
                // Return the last argument as a best-effort result, or Void.
                Ok(args.into_iter().last().unwrap_or(Value::Void))
            }
        }
    }

    /// Apply a Rust-style `"{} text {}"` format string to a list of values.
    /// First arg must be a string literal; subsequent args fill `{}` holes in order.
    /// If no format string is present, display all args separated by spaces.
    pub(crate) fn macro_format(args: &[Value], _line: usize) -> Result<String, Signal> {
        match args.first() {
            Some(Value::Str(fmt)) => {
                let fmt = fmt.clone();
                let mut result = String::new();
                let mut arg_iter = args.iter().skip(1);
                let mut chars = fmt.chars().peekable();
                while let Some(c) = chars.next() {
                    if c == '{' {
                        match chars.peek() {
                            Some('}') => {
                                chars.next(); // consume '}'
                                let val = arg_iter.next().unwrap_or(&Value::Nil);
                                result.push_str(&format!("{}", val));
                            }
                            Some(':') => {
                                // consume until matching '}'
                                let mut spec = String::new();
                                chars.next(); // consume ':'
                                for ch in chars.by_ref() {
                                    if ch == '}' { break; }
                                    spec.push(ch);
                                }
                                let val = arg_iter.next().unwrap_or(&Value::Nil);
                                // honour a small set of common format specifiers
                                let formatted = match spec.as_str() {
                                    "?" | "#?" => format!("{:?}", val),
                                    "x"  => if let Value::Int(n) = val { format!("{:x}", n) } else { format!("{}", val) },
                                    "X"  => if let Value::Int(n) = val { format!("{:X}", n) } else { format!("{}", val) },
                                    "b"  => if let Value::Int(n) = val { format!("{:b}", n) } else { format!("{}", val) },
                                    "o"  => if let Value::Int(n) = val { format!("{:o}", n) } else { format!("{}", val) },
                                    "e"  => if let Value::Float(f) = val { format!("{:e}", f) } else { format!("{}", val) },
                                    "E"  => if let Value::Float(f) = val { format!("{:E}", f) } else { format!("{}", val) },
                                    _ => format!("{}", val),   // ignore unknown specifiers
                                };
                                result.push_str(&formatted);
                            }
                            Some('{') => {
                                // escaped `{{` → literal `{`
                                chars.next();
                                result.push('{');
                            }
                            _ => {
                                // not a hole — keep verbatim
                                result.push('{');
                            }
                        }
                    } else if c == '}' && chars.peek() == Some(&'}') {
                        chars.next(); // `}}` → `}`
                        result.push('}');
                    } else {
                        result.push(c);
                    }
                }
                Ok(result)
            }
            // No format string — display all args separated by spaces
            _ => Ok(args.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join(" ")),
        }
    }

}
