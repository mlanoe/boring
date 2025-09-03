use super::*;
use crate::ast::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::cell::RefCell;
use std::rc::Rc;
use std::fmt;

impl Interpreter {
    pub fn eval_expr(&mut self, expr: &Expr, env: EnvRef) -> Eval {
        let line = expr.line;
        match &expr.kind {
            ExprKind::Int(n) => Ok(Value::Int(*n)),
            ExprKind::Float(f) => Ok(Value::Float(*f)),
            ExprKind::Str(s) => Ok(Value::Str(s.clone())),
            ExprKind::Bool(b) => Ok(Value::Bool(*b)),
            ExprKind::Nil => Ok(Value::Nil),
            ExprKind::Void => Ok(Value::Void),

            ExprKind::StringInterp(segments) => {
                let mut result = String::new();
                for seg in segments {
                    match seg {
                        StringSegment::Lit(s) => result.push_str(s),
                        StringSegment::Expr(e) => {
                            let v = self.eval_expr(e, Rc::clone(&env))?;
                            let line = e.line;
                            result.push_str(&self.display_value(v, line)?);
                        }
                        StringSegment::FormattedExpr(e, fmt) => {
                            let v = self.eval_expr(e, Rc::clone(&env))?;
                            let formatted = Self::apply_format(v, fmt, line)?;
                            result.push_str(&formatted);
                        }
                    }
                }
                Ok(Value::Str(result))
            }

            ExprKind::Var(name) => {
                if let Some(val) = env.borrow().get(name) {
                    return Ok(val);
                }
                // If the name is a type alias pointing to a struct or enum, return
                // the underlying constructor so `Dog2("rex")` works when
                // `use Dog2 as Dog'stack` has been declared.
                // The qualifier (stack, auto, task, …) only matters for the Rust
                // transpiler — the interpreter creates the same value regardless.
                if let Some(alias_ty) = self.aliases.get(name).cloned() {
                    let resolved = self.resolve_type(&alias_ty);
                    if let Some(base_name) = Self::type_base_name(&resolved) {
                        if let Some(ctor) = self.global.borrow().get(&base_name) {
                            if matches!(ctor, Value::Struct { .. } | Value::EnumNamespace { .. }) {
                                return Ok(ctor);
                            }
                        }
                    }
                }
                // Implicit self: `fieldname` inside a method resolves to `self.fieldname`
                if let Some(self_val) = env.borrow().get("self") {
                    if let Value::Object(inner_rc) = &self_val {
                        if let Some((_, field_val)) = inner_rc.borrow().fields.iter().find(|(k, _)| k == name) {
                            return Ok(field_val.clone());
                        }
                    }
                }
                let candidates = env.borrow().all_names();
                let msg = match closest_name(name, &candidates) {
                    Some(s) => format!("undefined variable '{}' — did you mean '{}'?", name, s),
                    None    => format!("undefined variable '{}'", name),
                };
                Err(err(msg, line))
            }

            ExprKind::BinOp(op, lhs, rhs) => {
                self.eval_binop(op, lhs, rhs, env, line)
            }

            ExprKind::UnaryOp(op, expr) => {
                let val = self.eval_expr(expr, Rc::clone(&env))?;
                match op {
                    UnaryOp::Neg => match val {
                        Value::Int(n) => Ok(Value::Int(-n)),
                        Value::Float(f) => Ok(Value::Float(-f)),
                        Value::Object(ref inner_rc) => {
                            let type_name = inner_rc.borrow().type_name.clone();
                            if self.has_method(&type_name, "neg") {
                                let mut out_self = None;
                                self.call_method(val.clone(), "neg", vec![], line, &mut out_self)
                            } else {
                                Err(err(format!("cannot negate {}", val.type_name()), line))
                            }
                        }
                        other => Err(err(format!("cannot negate {}", other.type_name()), line)),
                    },
                    UnaryOp::Not => {
                        let b = self.expect_bool(val, line)?;
                        Ok(Value::Bool(!b))
                    }
                    UnaryOp::BitNot => match val {
                        Value::Int(n)  => Ok(Value::Int(!n)),
                        Value::Uint(n) => Ok(Value::Uint(!n)),
                        other => Err(err(format!("cannot bitwise-not {}", other.type_name()), line)),
                    },
                }
            }

            ExprKind::Assign(target, value) => {
                let val = self.eval_expr(value, Rc::clone(&env))?;
                self.assign(target, val.clone(), Rc::clone(&env), line)?;
                Ok(val)
            }

            ExprKind::Field(obj_expr, field) => {
                // Type-level access: `Counter.MAX` or `Counter.count`
                if let ExprKind::Var(type_name) = &obj_expr.kind {
                    if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        let is_struct = self.global.borrow().get(type_name)
                            .map(|v| matches!(v, Value::Struct { .. }))
                            .unwrap_or(false);
                        if is_struct {
                            let key = format!("{}::{}", type_name, field);
                            if let Some(val) = self.type_var_store.get(&key).cloned() {
                                return Ok(val);
                            }
                            return Err(err(
                                format!("'{}' has no type var or type let '{}'", type_name, field),
                                line,
                            ));
                        }
                    }
                }
                let obj = self.eval_expr(obj_expr, Rc::clone(&env))?;
                self.get_field(obj, field, line)
            }

            ExprKind::Index(obj_expr, idx_expr) => {
                let obj = self.eval_expr(obj_expr, Rc::clone(&env))?;
                let idx = self.eval_expr(idx_expr, Rc::clone(&env))?;
                self.get_index(obj, idx, line)
            }

            ExprKind::Call(callee_expr, args) => {
                // channel/oneshot/broadcast/watch without type args: return (Sender, Receiver) pair
                if let ExprKind::Var(name) = &callee_expr.kind {
                    if matches!(name.as_str(), "channel" | "oneshot" | "broadcast" | "watch") {
                        let buf = Rc::new(RefCell::new(VecDeque::new()));
                        let closed = Rc::new(RefCell::new(false));
                        let sender = Value::Channel { buf: Rc::clone(&buf), closed: Rc::clone(&closed), is_sender: true };
                        let receiver = Value::Channel { buf, closed, is_sender: false };
                        return Ok(Value::Tuple(vec![sender, receiver]));
                    }
                    // timeout(dur, fut) — interpreter: skip duration, just evaluate the future
                    if name.as_str() == "timeout" {
                        if let Some(fut_arg) = args.get(1) {
                            return self.eval_expr(&fut_arg.value, Rc::clone(&env));
                        }
                        return Ok(Value::Nil);
                    }
                    // json(v) — interpreter stub: convert value to its debug string representation
                    if name.as_str() == "json" {
                        if let Some(arg) = args.first() {
                            let v = self.eval_expr(&arg.value, Rc::clone(&env))?;
                            return Ok(Value::Str(format!("{:?}", v).into()));
                        }
                        return Ok(Value::Str("null".into()));
                    }
                    // print / write / log-level — use `as string:` conversions for Object args
                    if matches!(name.as_str(), "print" | "write" | "error" | "warn" | "info" | "debug" | "trace") {
                        let arg_vals = self.eval_args(args, Rc::clone(&env))?;
                        return self.call_display_builtin(name, &arg_vals, line);
                    }
                }
                let callee = self.eval_expr(callee_expr, Rc::clone(&env))?;
                // Check for double-use of owned args before evaluating
                if let Value::Fn { ref decl, .. } = callee {
                    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                    for (param, arg) in decl.params.iter().zip(args.iter()) {
                        if param.owned {
                            if let ExprKind::Var(name) = &arg.value.kind {
                                if !seen.insert(name.clone()) {
                                    return Err(err(format!("'{}' moved twice in the same call", name), line));
                                }
                            }
                        }
                    }
                }
                for arg in args.iter() {
                    Self::check_no_owned_extract(&arg.value, &env, line)?;
                }
                let arg_vals = self.eval_args(args, Rc::clone(&env))?;
                let result = self.call_value(callee.clone(), arg_vals, line, false)?;
                // Invalidate owned param sources after successful call
                if let Value::Fn { ref decl, .. } = callee {
                    for (param, arg) in decl.params.iter().zip(args.iter()) {
                        if param.owned {
                            if let ExprKind::Var(name) = &arg.value.kind {
                                env.borrow_mut().invalidate(name);
                            }
                        }
                    }
                }
                Ok(result)
            }

            ExprKind::MethodCall(obj_expr, method, args) => {
                // Built-in `fs` module namespace — intercept before evaluating the receiver
                // so that `fs` does not need to be defined as a variable.
                if let ExprKind::Var(v) = &obj_expr.kind {
                    if v == "fs" {
                        let arg_vals = self.eval_args(args, Rc::clone(&env))?;
                        return self.call_fs_method(method, arg_vals, line);
                    }
                }

                // Type-level call: `Counter.zero()` or `Counter.set_count(v)`
                if let ExprKind::Var(type_name) = &obj_expr.kind {
                    if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        let struct_val = self.global.borrow().get(type_name);
                        if let Some(Value::Struct { decl, captured }) = struct_val {
                            let tm = decl.type_methods.iter().find(|m| m.name == *method).cloned();
                            if let Some(type_method) = tm {
                                let arg_vals = self.eval_args(args, Rc::clone(&env))?;
                                return self.call_type_method(&decl.name.clone(), &type_method, arg_vals, Rc::clone(&captured), line);
                            }
                            return Err(err(
                                format!("'{}' has no type method '{}'", type_name, method),
                                line,
                            ));
                        }
                    }
                }
                let obj = self.eval_expr(obj_expr, Rc::clone(&env))?;
                // Enforce: mutating method cannot be called on an immutable (let) binding
                // Built-in non-mutating methods (e.g. `upgrade`) bypass this check.
                const BUILTIN_NON_MUTATING: &[&str] = &["upgrade"];
                if let ExprKind::Var(binding_name) = &obj_expr.kind {
                    if !BUILTIN_NON_MUTATING.contains(&method.as_str()) {
                        if let Value::Object(inner_rc) = &obj {
                            let type_name = inner_rc.borrow().type_name.clone();
                            let is_mutating = {
                                let g = self.global.borrow();
                                if let Some(Value::Struct { ref decl, .. }) = g.get(&type_name) {
                                    // `task` methods take Arc<Self> — not &mut self — so never count as mutating.
                                    decl.methods.iter().find(|m| m.name == *method).map(|m| m.mutating && !m.task)
                                        .unwrap_or(true)
                                } else { true }
                            };
                            if is_mutating && !env.borrow().is_mutable(binding_name) {
                                return Err(err(
                                    format!("cannot call mutating method '{}' on let binding '{}'", method, binding_name),
                                    line,
                                ));
                            }
                            if is_mutating && env.borrow().is_shared(binding_name) {
                                return Err(err(
                                    format!("cannot call mutating method '{}' on shared binding '{}' — use T'actor for interior mutability", method, binding_name),
                                    line,
                                ));
                            }
                        }
                    }
                }
                // Check for double-use of owned args before evaluating
                if let Value::Object(inner_rc) = &obj {
                    let type_name = inner_rc.borrow().type_name.clone();
                    let decl_opt = {
                        let g = self.global.borrow();
                        if let Some(Value::Struct { ref decl, .. }) = g.get(&type_name) {
                            decl.methods.iter().find(|m| m.name == *method).cloned()
                        } else { None }
                    };
                    if let Some(fn_decl) = decl_opt {
                        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                        for (param, arg) in fn_decl.params.iter().zip(args.iter()) {
                            if param.owned {
                                if let ExprKind::Var(name) = &arg.value.kind {
                                    if !seen.insert(name.clone()) {
                                        return Err(err(format!("'{}' moved twice in the same call", name), line));
                                    }
                                }
                            }
                        }
                    }
                }
                let arg_vals = self.eval_args(args, Rc::clone(&env))?;
                let mut modified_self: Option<Value> = None;
                let result = self.call_method(obj.clone(), method, arg_vals, line, &mut modified_self)?;
                // Write back modified self to source variable (force: mut method on let binding is OK)
                if let Some(new_obj) = modified_self {
                    match &obj_expr.kind {
                        ExprKind::Var(name) => { env.borrow_mut().force_set(name, new_obj); }
                        _ => { let _ = self.assign(obj_expr, new_obj, Rc::clone(&env), line); }
                    }
                }
                // Collect owned-param source names before any borrow of global
                let mut to_invalidate: Vec<String> = Vec::new();
                if let Value::Object(inner_rc) = &obj {
                    let type_name = inner_rc.borrow().type_name.clone();
                    let names: Option<Vec<String>> = {
                        let g = self.global.borrow();
                        if let Some(Value::Struct { ref decl, .. }) = g.get(&type_name) {
                            if let Some(fn_decl) = decl.methods.iter().find(|m| m.name == *method) {
                                Some(fn_decl.params.iter().zip(args.iter()).filter_map(|(param, arg)| {
                                    if param.owned {
                                        if let ExprKind::Var(name) = &arg.value.kind {
                                            return Some(name.clone());
                                        }
                                    }
                                    None
                                }).collect())
                            } else { None }
                        } else { None }
                    }; // global borrow dropped here
                    if let Some(names) = names {
                        to_invalidate = names;
                    }
                }
                for name in to_invalidate {
                    env.borrow_mut().invalidate(&name);
                }
                Ok(result)
            }

            ExprKind::TryElse(try_expr, default_expr) => {
                // Evaluate inner; if exception or Err(e), evaluate default.
                // Also handles `def Result` functions that return Ok(v)/Err(e) enum variants:
                //   Ok(v)  → unwrap to the inner value v
                //   Err(_) → evaluate default (nil for `try?`)
                match self.eval_expr(try_expr, Rc::clone(&env)) {
                    Ok(Value::EnumVariant { ref type_name, ref variant, ref fields })
                        if type_name == "Result" && variant == "Ok" =>
                    {
                        Ok(fields.first().cloned().unwrap_or(Value::Nil))
                    }
                    Ok(Value::EnumVariant { ref type_name, ref variant, .. })
                        if type_name == "Result" && variant == "Err" =>
                    {
                        self.eval_expr(default_expr, env)
                    }
                    Ok(v) => Ok(v),
                    Err(Signal::Exception(_)) => self.eval_expr(default_expr, env),
                    Err(other) => Err(other),
                }
            }

            ExprKind::TryElseBlock(try_stmts, else_stmts) => {
                // Multi-line try/else block expression.
                // Execute try body; if it throws (or returns Err), execute else body with
                // `error` bound to the **original thrown value** in the else scope.
                //
                // `error` keeps its original type so the else body can pattern-match on it:
                //   try risky() else:
                //       match error:
                //           MyError.NotFound: "not found"
                //           _: "other: {error}"
                //
                // String interpolation `{error}` still works because the interpreter's
                // `display_value` falls back to the Value's Display for non-Object types,
                // and `as string:` conversions are honoured for struct/enum values.
                let try_env = Env::child(Rc::clone(&env));
                let result = self.eval_block_as_expr(try_stmts, try_env);

                match result {
                    Ok(v) => {
                        // If the block returned Err(e) directly, fall through to else body
                        // with the inner error value bound to `error`.
                        if let Value::EnumVariant { ref type_name, ref variant, ref fields } = v {
                            if type_name == "Result" && variant == "Err" {
                                let err_val = fields.first().cloned().unwrap_or(Value::Nil);
                                let else_env = Env::child(Rc::clone(&env));
                                else_env.borrow_mut().define("error", err_val);
                                return self.eval_block_as_expr(else_stmts, else_env);
                            }
                        }
                        // Unwrap Ok(v) enum variants produced by `def Result` functions.
                        if let Value::EnumVariant { ref type_name, ref variant, ref fields } = v {
                            if type_name == "Result" && variant == "Ok" {
                                return Ok(fields.first().cloned().unwrap_or(Value::Nil));
                            }
                        }
                        Ok(v)
                    }
                    Err(Signal::Exception(err_val)) => {
                        // Bind `error` to the original thrown value, not its string form.
                        // This allows `match error: MyEnum.Variant: …` in the else body.
                        let else_env = Env::child(Rc::clone(&env));
                        else_env.borrow_mut().define("error", err_val);
                        self.eval_block_as_expr(else_stmts, else_env)
                    }
                    Err(other) => Err(other),
                }
            }

            ExprKind::Else(expr, default) => {
                let val = self.eval_expr(expr, Rc::clone(&env))?;
                match val {
                    Value::Nil => self.eval_expr(default, env),
                    other => Ok(other),
                }
            }

            ExprKind::OptionalField(obj_expr, field) => {
                let obj = self.eval_expr(obj_expr, Rc::clone(&env))?;
                match obj {
                    Value::Nil => Ok(Value::Nil),
                    other => self.get_field(other, field, line),
                }
            }

            ExprKind::OptionalMethodCall(obj_expr, method, args) => {
                let obj = self.eval_expr(obj_expr, Rc::clone(&env))?;
                match obj {
                    Value::Nil => Ok(Value::Nil),
                    other => {
                        let arg_vals = self.eval_args(args, Rc::clone(&env))?;
                        let mut _out = None;
                        self.call_method(other, method, arg_vals, line, &mut _out)
                    }
                }
            }


            ExprKind::Array(elems) => {
                let mut vals = Vec::new();
                for e in elems {
                    vals.push(self.eval_expr(e, Rc::clone(&env))?);
                }
                Ok(Value::Array(vals))
            }

            ExprKind::Tuple(elems) => {
                let mut vals = Vec::new();
                for e in elems {
                    vals.push(self.eval_expr(e, Rc::clone(&env))?);
                }
                Ok(Value::Tuple(vals))
            }

            ExprKind::Dict(pairs) => {
                let mut result = Vec::new();
                for (k, v) in pairs {
                    let kv = self.eval_expr(k, Rc::clone(&env))?;
                    let vv = self.eval_expr(v, Rc::clone(&env))?;
                    result.push((kv, vv));
                }
                Ok(Value::Dict(result))
            }

            ExprKind::Set(elems) => {
                let mut vals: Vec<Value> = Vec::new();
                for e in elems {
                    let v = self.eval_expr(e, Rc::clone(&env))?;
                    if !vals.contains(&v) {
                        vals.push(v);
                    }
                }
                Ok(Value::Set(vals))
            }

            ExprKind::DotIdent(name) => {
                // Enum shorthand: .Red — search global scope for an enum with this variant
                let global = self.global.borrow();
                let mut found: Option<Value> = None;
                let mut ambiguous = false;
                for val in global.vars.values() {
                    if let Value::EnumNamespace { variants, .. } = val {
                        if let Some(variant_val) = variants.get(name.as_str()) {
                            if found.is_some() {
                                ambiguous = true;
                                break;
                            }
                            found = Some(variant_val.clone());
                        }
                    }
                }
                drop(global);
                if ambiguous {
                    return Err(err(
                        format!("ambiguous dot-prefix '.{}': multiple enums define this variant; use EnumName.{} instead", name, name),
                        line,
                    ));
                }
                if let Some(v) = found {
                    return Ok(v);
                }
                // Fallback: return as a string marker (old behavior for compatibility)
                Ok(Value::Str(format!(".{}", name)))
            }

            ExprKind::Range { start, end, inclusive } => {
                let s = self.eval_expr(start, Rc::clone(&env))?;
                let e = self.eval_expr(end, Rc::clone(&env))?;
                match (s, e) {
                    (Value::Int(a), Value::Int(b)) => Ok(Value::Range { start: a, end: b, inclusive: *inclusive }),
                    (a, b) => Err(err(format!("range requires Int bounds, got {} and {}", a.type_name(), b.type_name()), line)),
                }
            }

            ExprKind::Cast(expr, ty) => {
                let val = self.eval_expr(expr, Rc::clone(&env))?;
                self.cast_value(val, ty, line)
            }

            ExprKind::Closure(params, _ret_ty, body, _throws, _task) => {
                Ok(Value::Closure {
                    params: params.clone(),
                    body: body.clone(),
                    captured: env,
                })
            }

            ExprKind::If(if_stmt) => {
                // If as expression — execute and return last value or Nil
                // We capture the return via the statement executor
                let child = Env::child(Rc::clone(&env));
                // Run if as stmt but capture result
                let val = self.eval_if_expr(if_stmt, child)?;
                Ok(val)
            }

            ExprKind::Match(match_stmt) => {
                self.eval_match_expr(match_stmt, env)
            }

            ExprKind::Block(stmts) => {
                let child = Env::child(Rc::clone(&env));
                self.eval_block_as_expr(stmts, child)
            }

            ExprKind::Do(stmts) => {
                // `do:` — own scope + own defer frame; last expression is the value.
                // `defer` statements within the block run when the block exits (not the function).
                // `return`/`break`/`continue` propagate outward normally.
                let child = Env::child(Rc::clone(&env));
                self.defer_stack.push(Vec::new());
                // Execute all but the last statement first.
                let pre = self.exec_all_but_last(stmts, Rc::clone(&child));
                // Evaluate the tail first (may register more defers).
                let val = match pre {
                    Ok(()) => {
                        if let Some(last) = stmts.last() {
                            self.eval_tail_stmt(last, Rc::clone(&child))
                        } else {
                            Ok(Value::Nil)
                        }
                    }
                    Err(other) => Err(other),
                };
                // Run deferred blocks in LIFO order AFTER the block body has finished.
                if let Some(frame) = self.defer_stack.pop() {
                    for deferred in frame.into_iter().rev() {
                        let _ = self.exec_block(&deferred, Rc::clone(&child));
                    }
                }
                val
            }

            ExprKind::Loop(loop_stmt) => {
                self.eval_loop(loop_stmt, env)
            }

            ExprKind::Task(inner) => {
                // Collect owned vars captured by this task so we can invalidate them after
                let owned_captures = Self::collect_owned_task_captures(inner, &env);
                Self::check_task_captures(inner, &env, line)?;
                let val = self.eval_expr(inner, Rc::clone(&env))?;
                // Invalidate owned captures — the task is now the sole owner
                for name in &owned_captures {
                    env.borrow_mut().invalidate(name);
                }
                Ok(Value::Future(Box::new(val)))
            }

            ExprKind::MacroCall { name, args } => {
                // Evaluate all arguments eagerly
                let mut arg_vals = Vec::new();
                for a in args {
                    arg_vals.push(self.eval_expr(a, Rc::clone(&env))?);
                }
                self.call_macro(name, arg_vals, line)
            }

            ExprKind::Pipe(lhs, method, args) => {
                let obj = self.eval_expr(lhs, Rc::clone(&env))?;
                let mut extra_args = Vec::new();
                for a in args {
                    extra_args.push(self.eval_expr(&a.value, Rc::clone(&env))?);
                }
                // Try as a free function first, then fall back to method call on the object
                let func = self.global.borrow().get(method);
                if let Some(callee) = func {
                    let mut all_args = vec![obj];
                    all_args.extend(extra_args);
                    self.call_value(callee, all_args, line, false)
                } else {
                    let mut out_self = None;
                    self.call_method(obj, method, extra_args, line, &mut out_self)
                }
            }

            ExprKind::JoinAll(exprs) => {
                // In the interpreter, JoinAll evaluates futures sequentially (no real parallelism).
                let mut results = Vec::new();
                for e in exprs {
                    results.push(self.eval_expr(e, Rc::clone(&env))?);
                }
                Ok(Value::Tuple(results))
            }

            ExprKind::GenericCall(callee, _type_args, args) => {
                // In the interpreter, type args are erased — just evaluate as a regular call.
                // Special built-ins that need type info (like `channel`) return a pair of arrays
                // as a synchronous simulation: (sender_items, receiver_items) backed by a Vec.
                if let ExprKind::Var(name) = &callee.kind {
                    if matches!(name.as_str(), "channel" | "oneshot" | "broadcast" | "watch") {
                        let buf = Rc::new(RefCell::new(VecDeque::new()));
                        let closed = Rc::new(RefCell::new(false));
                        let sender = Value::Channel { buf: Rc::clone(&buf), closed: Rc::clone(&closed), is_sender: true };
                        let receiver = Value::Channel { buf, closed, is_sender: false };
                        return Ok(Value::Tuple(vec![sender, receiver]));
                    }
                    if name.as_str() == "timeout" {
                        if let Some(fut_arg) = args.get(1) {
                            return self.eval_expr(&fut_arg.value, Rc::clone(&env));
                        }
                        return Ok(Value::Nil);
                    }
                    // from_json<T>(s) — interpreter stub: return the string as-is (no deserialization)
                    if name.as_str() == "fromJson" {
                        if let Some(arg) = args.first() {
                            return self.eval_expr(&arg.value, Rc::clone(&env));
                        }
                        return Ok(Value::Nil);
                    }
                }
                // Generic call with no special handling: evaluate callee and call it.
                let callee_val = self.eval_expr(callee, Rc::clone(&env))?;
                let mut evaled_args = Vec::new();
                for a in args {
                    evaled_args.push(self.eval_expr(&a.value, Rc::clone(&env))?);
                }
                self.call_value(callee_val, evaled_args, line, false)
            }
        }
    }

    pub(crate) fn eval_if_expr(&mut self, s: &IfStmt, env: EnvRef) -> Eval {
        for (cond, body) in &s.branches {
            let val = self.eval_expr(cond, Rc::clone(&env))?;
            let b = self.expect_bool(val, cond.line)?;
            if b {
                let child = Env::child(Rc::clone(&env));
                return self.eval_block_as_expr(body, child);
            }
        }
        if let Some(else_body) = &s.else_body {
            let child = Env::child(Rc::clone(&env));
            return self.eval_block_as_expr(else_body, child);
        }
        Ok(Value::Nil)
    }

    pub(crate) fn eval_match_expr(&mut self, s: &MatchStmt, env: EnvRef) -> Eval {
        let subject = self.eval_expr(&s.subject, Rc::clone(&env))?;
        'arms: for arm in &s.arms {
            for pattern in &arm.patterns {
                let mut bindings = HashMap::new();
                if self.match_pattern(pattern, &subject, &mut bindings) {
                    let child = Env::child(Rc::clone(&env));
                    for (k, v) in bindings {
                        child.borrow_mut().define(&k, v);
                    }
                    // Evaluate optional guard in the child env (bindings already in scope)
                    if let Some(guard_expr) = &arm.guard {
                        let guard_val = self.eval_expr(guard_expr, Rc::clone(&child))?;
                        if !self.expect_bool(guard_val, guard_expr.line)? {
                            continue 'arms;
                        }
                    }
                    return match &arm.body {
                        MatchBody::Expr(e) => self.eval_expr(e, child),
                        MatchBody::Block(stmts) => self.eval_block_as_expr(stmts, child),
                    };
                }
            }
        }
        Ok(Value::Nil)
    }

    pub(crate) fn eval_block_as_expr(&mut self, stmts: &[Stmt], env: EnvRef) -> Eval {
        let mut last = Value::Nil;
        for (i, stmt) in stmts.iter().enumerate() {
            if i == stmts.len() - 1 {
                // For the last statement, extract its value when possible so that
                // `if cond: a else b` and `match … { … }` work as tail expressions.
                match stmt {
                    Stmt::Expr(e) => {
                        let v = self.eval_expr(e, Rc::clone(&env))?;
                        // Assignments are side-effects; they should not become the implicit
                        // return value of a block/function body.
                        if !matches!(e.kind, ExprKind::Assign(..)) {
                            last = v;
                        }
                        continue;
                    }
                    Stmt::If(s) => {
                        return self.eval_if_expr(s, Rc::clone(&env));
                    }
                    Stmt::Match(s) => {
                        return self.eval_match_expr(s, env);
                    }
                    _ => {}
                }
            }
            match self.exec_stmt(stmt, Rc::clone(&env)) {
                Ok(()) => {}
                Err(Signal::Return(v)) => return Ok(v),
                Err(other) => return Err(other),
            }
        }
        Ok(last)
    }

    /// Execute every statement in `stmts` except the very last one.
    /// Used when we need to run deferred blocks BEFORE the tail expression so that
    /// defers that mutate local variables are visible to the implicit return value.
    /// Any control-flow signal (Return, Break, Continue, error) propagates immediately.
    pub(crate) fn exec_all_but_last(&mut self, stmts: &[Stmt], env: EnvRef) -> Result<(), Signal> {
        let n = stmts.len().saturating_sub(1);
        for stmt in &stmts[..n] {
            self.exec_stmt(stmt, Rc::clone(&env))?;
        }
        Ok(())
    }

    /// Evaluate the last statement of a function / block body as a tail expression.
    /// Mirrors the special-case logic inside `eval_block_as_expr` for the final stmt.
    pub(crate) fn eval_tail_stmt(&mut self, stmt: &Stmt, env: EnvRef) -> Eval {
        match stmt {
            Stmt::Expr(e) => {
                let v = self.eval_expr(e, Rc::clone(&env))?;
                if matches!(e.kind, ExprKind::Assign(..)) {
                    Ok(Value::Nil)
                } else {
                    Ok(v)
                }
            }
            Stmt::If(s)    => self.eval_if_expr(s, env),
            Stmt::Match(s) => self.eval_match_expr(s, env),
            _ => {
                self.exec_stmt(stmt, Rc::clone(&env))?;
                Ok(Value::Nil)
            }
        }
    }

    /// Like `eval_block_as_expr` but transparent to control-flow signals.
    /// Used for `do:` blocks: `return`/`break`/`continue` propagate to the
    /// enclosing function/loop rather than being captured by the block itself.
    pub(crate) fn eval_do_block(&mut self, stmts: &[Stmt], env: EnvRef) -> Eval {
        let mut last = Value::Nil;
        for (i, stmt) in stmts.iter().enumerate() {
            if i == stmts.len() - 1 {
                match stmt {
                    Stmt::Expr(e) => {
                        let v = self.eval_expr(e, Rc::clone(&env))?;
                        if !matches!(e.kind, ExprKind::Assign(..)) {
                            last = v;
                        }
                        continue;
                    }
                    Stmt::If(s) => {
                        return self.eval_if_expr(s, Rc::clone(&env));
                    }
                    Stmt::Match(s) => {
                        return self.eval_match_expr(s, env);
                    }
                    _ => {}
                }
            }
            // All signals (Return, Break, Continue, RuntimeError) propagate outward.
            self.exec_stmt(stmt, Rc::clone(&env))?;
        }
        Ok(last)
    }

    pub(crate) fn eval_binop(&mut self, op: &BinOp, lhs: &Expr, rhs: &Expr, env: EnvRef, line: usize) -> Eval {
        // Short-circuit for And/Or
        if *op == BinOp::And {
            let l = self.eval_expr(lhs, Rc::clone(&env))?;
            let b = self.expect_bool(l, line)?;
            if !b { return Ok(Value::Bool(false)); }
            let r = self.eval_expr(rhs, env)?;
            let b2 = self.expect_bool(r, line)?;
            return Ok(Value::Bool(b2));
        }
        if *op == BinOp::Or {
            let l = self.eval_expr(lhs, Rc::clone(&env))?;
            let b = self.expect_bool(l, line)?;
            if b { return Ok(Value::Bool(true)); }
            let r = self.eval_expr(rhs, env)?;
            let b2 = self.expect_bool(r, line)?;
            return Ok(Value::Bool(b2));
        }

        let l = self.eval_expr(lhs, Rc::clone(&env))?;
        let r = self.eval_expr(rhs, env)?;

        match op {
            BinOp::Add => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                (Value::Uint(a), Value::Uint(b)) => Ok(Value::Uint(a.wrapping_add(b))),
                (Value::Uint(a), Value::Int(b)) => {
                    if b < 0 { Err(err("cannot add negative Int to Uint", line)) }
                    else { Ok(Value::Uint(a.wrapping_add(b as u64))) }
                }
                (Value::Int(a), Value::Uint(b)) => {
                    if a < 0 { Err(err("cannot add negative Int to Uint", line)) }
                    else { Ok(Value::Uint((a as u64).wrapping_add(b))) }
                }
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 + b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + b as f64)),
                (Value::Str(a), Value::Str(b)) => Ok(Value::Str(a + &b)),
                (Value::Array(mut a), Value::Array(b)) => { a.extend(b); Ok(Value::Array(a)) }
                (a, b) => {
                    if let Some(result) = self.try_operator_method(&a, "add", b.clone(), line)? {
                        Ok(result)
                    } else {
                        Err(err(format!("cannot add {} and {}", a.type_name(), b.type_name()), line))
                    }
                }
            },
            BinOp::Sub => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
                (Value::Uint(a), Value::Uint(b)) => {
                    if b > a { Err(err("uint subtraction underflow", line)) }
                    else { Ok(Value::Uint(a - b)) }
                }
                (Value::Uint(a), Value::Int(b)) => {
                    if b < 0 { Ok(Value::Uint(a.wrapping_add((-b) as u64))) }
                    else if (b as u64) > a { Err(err("uint subtraction underflow", line)) }
                    else { Ok(Value::Uint(a - b as u64)) }
                }
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 - b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - b as f64)),
                (a, b) => {
                    if let Some(result) = self.try_operator_method(&a, "sub", b.clone(), line)? {
                        Ok(result)
                    } else {
                        Err(err(format!("cannot subtract {} and {}", a.type_name(), b.type_name()), line))
                    }
                }
            },
            BinOp::Mul => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
                (Value::Uint(a), Value::Uint(b)) => Ok(Value::Uint(a.wrapping_mul(b))),
                (Value::Uint(a), Value::Int(b)) => {
                    if b < 0 { Err(err("cannot multiply Uint by negative Int", line)) }
                    else { Ok(Value::Uint(a.wrapping_mul(b as u64))) }
                }
                (Value::Int(a), Value::Uint(b)) => {
                    if a < 0 { Err(err("cannot multiply Uint by negative Int", line)) }
                    else { Ok(Value::Uint((a as u64).wrapping_mul(b))) }
                }
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 * b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * b as f64)),
                (a, b) => {
                    if let Some(result) = self.try_operator_method(&a, "mul", b.clone(), line)? {
                        Ok(result)
                    } else {
                        Err(err(format!("cannot multiply {} and {}", a.type_name(), b.type_name()), line))
                    }
                }
            },
            BinOp::Div => match (l, r) {
                (Value::Int(a), Value::Int(b)) => {
                    if b == 0 { Err(err("division by zero", line)) } else { Ok(Value::Int(a / b)) }
                }
                (Value::Uint(a), Value::Uint(b)) => {
                    if b == 0 { Err(err("division by zero", line)) } else { Ok(Value::Uint(a / b)) }
                }
                (Value::Uint(a), Value::Int(b)) => {
                    if b == 0 { Err(err("division by zero", line)) }
                    else if b < 0 { Err(err("cannot divide Uint by negative Int", line)) }
                    else { Ok(Value::Uint(a / b as u64)) }
                }
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 / b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / b as f64)),
                (a, b) => {
                    if let Some(result) = self.try_operator_method(&a, "div", b.clone(), line)? {
                        Ok(result)
                    } else {
                        Err(err(format!("cannot divide {} and {}", a.type_name(), b.type_name()), line))
                    }
                }
            },
            BinOp::Rem => match (l, r) {
                (Value::Int(a), Value::Int(b)) => {
                    if b == 0 { Err(err("remainder by zero", line)) } else { Ok(Value::Int(a % b)) }
                }
                (Value::Uint(a), Value::Uint(b)) => {
                    if b == 0 { Err(err("remainder by zero", line)) } else { Ok(Value::Uint(a % b)) }
                }
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a % b)),
                (Value::Int(a),   Value::Float(b)) => Ok(Value::Float((a as f64) % b)),
                (Value::Float(a), Value::Int(b))   => Ok(Value::Float(a % (b as f64))),
                (a, b) => {
                    if let Some(result) = self.try_operator_method(&a, "rem", b.clone(), line)? {
                        Ok(result)
                    } else {
                        Err(err(format!("cannot remainder {} and {}", a.type_name(), b.type_name()), line))
                    }
                }
            },
            BinOp::Eq => {
                if let Some(result) = self.try_operator_method(&l, "eq", r.clone(), line)? {
                    Ok(result)
                } else {
                    Ok(Value::Bool(l == r))
                }
            }
            BinOp::RefEq => {
                // Reference equality — bypasses user-defined eq, always compares identity
                let result = match (&l, &r) {
                    (Value::Object(a), Value::Object(b)) => Rc::ptr_eq(a, b),
                    _ => l == r,   // primitives have value semantics, identity == equality
                };
                Ok(Value::Bool(result))
            }
            BinOp::NotEq => {
                if let Some(result) = self.try_operator_method(&l, "ne", r.clone(), line)? {
                    Ok(result)
                } else if let Some(eq_result) = self.try_operator_method(&l, "eq", r.clone(), line)? {
                    // Fall back to !eq if ne not defined
                    match eq_result {
                        Value::Bool(b) => Ok(Value::Bool(!b)),
                        other => Ok(Value::Bool(other != Value::Bool(true))),
                    }
                } else {
                    Ok(Value::Bool(l != r))
                }
            }
            BinOp::Lt => {
                if let Some(result) = self.try_operator_method(&l, "lt", r.clone(), line)? {
                    Ok(result)
                } else {
                    self.compare_values(l, r, |o| o == std::cmp::Ordering::Less, line)
                }
            }
            BinOp::Gt => {
                if let Some(result) = self.try_operator_method(&l, "gt", r.clone(), line)? {
                    Ok(result)
                } else {
                    self.compare_values(l, r, |o| o == std::cmp::Ordering::Greater, line)
                }
            }
            BinOp::LtEq => {
                if let Some(result) = self.try_operator_method(&l, "le", r.clone(), line)? {
                    Ok(result)
                } else {
                    self.compare_values(l, r, |o| o != std::cmp::Ordering::Greater, line)
                }
            }
            BinOp::GtEq => {
                if let Some(result) = self.try_operator_method(&l, "ge", r.clone(), line)? {
                    Ok(result)
                } else {
                    self.compare_values(l, r, |o| o != std::cmp::Ordering::Less, line)
                }
            }
            BinOp::And | BinOp::Or => unreachable!(),
            BinOp::BitAnd => match (l, r) {
                (Value::Int(a),  Value::Int(b))  => Ok(Value::Int(a & b)),
                (Value::Uint(a), Value::Uint(b)) => Ok(Value::Uint(a & b)),
                (Value::Uint(a), Value::Int(b))  => Ok(Value::Uint(a & b as u64)),
                (Value::Int(a),  Value::Uint(b)) => Ok(Value::Int(a & b as i64)),
                (a, b) => Err(err(format!("cannot bitwise-and {} and {}", a.type_name(), b.type_name()), line)),
            },
            BinOp::BitOr => match (l, r) {
                (Value::Int(a),  Value::Int(b))  => Ok(Value::Int(a | b)),
                (Value::Uint(a), Value::Uint(b)) => Ok(Value::Uint(a | b)),
                (Value::Uint(a), Value::Int(b))  => Ok(Value::Uint(a | b as u64)),
                (Value::Int(a),  Value::Uint(b)) => Ok(Value::Int(a | b as i64)),
                (a, b) => Err(err(format!("cannot bitwise-or {} and {}", a.type_name(), b.type_name()), line)),
            },
            BinOp::BitXor => match (l, r) {
                (Value::Int(a),  Value::Int(b))  => Ok(Value::Int(a ^ b)),
                (Value::Uint(a), Value::Uint(b)) => Ok(Value::Uint(a ^ b)),
                (Value::Uint(a), Value::Int(b))  => Ok(Value::Uint(a ^ b as u64)),
                (Value::Int(a),  Value::Uint(b)) => Ok(Value::Int(a ^ b as i64)),
                (a, b) => Err(err(format!("cannot bitwise-xor {} and {}", a.type_name(), b.type_name()), line)),
            },
            BinOp::Shl => match (l, r) {
                (Value::Int(a),  Value::Int(b))  if b >= 0 => Ok(Value::Int(a.wrapping_shl(b as u32))),
                (Value::Uint(a), Value::Int(b))  if b >= 0 => Ok(Value::Uint(a.wrapping_shl(b as u32))),
                (Value::Uint(a), Value::Uint(b)) => Ok(Value::Uint(a.wrapping_shl(b as u32))),
                (_, Value::Int(b)) if b < 0 => Err(err("shift amount cannot be negative", line)),
                (a, b) => Err(err(format!("cannot shift {} by {}", a.type_name(), b.type_name()), line)),
            },
            BinOp::Shr => match (l, r) {
                (Value::Int(a),  Value::Int(b))  if b >= 0 => Ok(Value::Int(a.wrapping_shr(b as u32))),
                (Value::Uint(a), Value::Int(b))  if b >= 0 => Ok(Value::Uint(a.wrapping_shr(b as u32))),
                (Value::Uint(a), Value::Uint(b)) => Ok(Value::Uint(a.wrapping_shr(b as u32))),
                (_, Value::Int(b)) if b < 0 => Err(err("shift amount cannot be negative", line)),
                (a, b) => Err(err(format!("cannot shift {} by {}", a.type_name(), b.type_name()), line)),
            },
            BinOp::Is => {
                // Case 1: `x is nil` — nil check
                if matches!(r, Value::Nil) {
                    return Ok(Value::Bool(matches!(l, Value::Nil)));
                }
                // Case 2: `x is TypeName` — type conformance check
                if let Value::Struct { ref decl, .. } = r {
                    let type_name = l.type_name();
                    // Direct type match
                    if type_name == decl.name { return Ok(Value::Bool(true)); }
                    // Trait conformance
                    let result = self.object_conforms_to_trait(&l.type_name(), &decl.name);
                    return Ok(Value::Bool(result));
                }
                if let Value::EnumNamespace { ref name, .. } = r {
                    return Ok(Value::Bool(l.type_name() == *name));
                }
                // Case 3: `a is b` — reference identity for Objects, value equality otherwise
                let result = match (&l, &r) {
                    (Value::Object(a), Value::Object(b)) => Rc::ptr_eq(a, b),
                    _ => l == r,
                };
                Ok(Value::Bool(result))
            }
            BinOp::IsNot => {
                // Negate `is` — inline the same logic then negate
                let is_result: bool = if matches!(r, Value::Nil) {
                    matches!(l, Value::Nil)
                } else if let Value::Struct { ref decl, .. } = r {
                    let type_name = l.type_name();
                    if type_name == decl.name {
                        true
                    } else {
                        self.object_conforms_to_trait(&l.type_name(), &decl.name)
                    }
                } else if let Value::EnumNamespace { ref name, .. } = r {
                    l.type_name() == *name
                } else {
                    match (&l, &r) {
                        (Value::Object(a), Value::Object(b)) => Rc::ptr_eq(a, b),
                        _ => l == r,
                    }
                };
                Ok(Value::Bool(!is_result))
            }
        }
    }

    pub(crate) fn compare_values(&self, l: Value, r: Value, pred: impl Fn(std::cmp::Ordering) -> bool, line: usize) -> Eval {
        let ord = match (&l, &r) {
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::Uint(a), Value::Uint(b)) => a.cmp(b),
            // Cross Int/Uint comparison: promote both to i128 to handle all cases safely
            (Value::Uint(a), Value::Int(b)) => (*a as i128).cmp(&(*b as i128)),
            (Value::Int(a), Value::Uint(b)) => (*a as i128).cmp(&(*b as i128)),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
            (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
            (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(std::cmp::Ordering::Equal),
            (Value::Str(a), Value::Str(b)) => a.cmp(b),
            _ => return Err(err(format!("cannot compare {} and {}", l.type_name(), r.type_name()), line)),
        };
        Ok(Value::Bool(pred(ord)))
    }

    pub(crate) fn eval_args(&mut self, args: &[Arg], env: EnvRef) -> Result<Vec<Value>, Signal> {
        let mut vals = Vec::new();
        for arg in args {
            let v = self.eval_expr(&arg.value, Rc::clone(&env))?;
            if arg.spread {
                // `..expr` — expand all fields of the source struct as labeled args.
                // Explicit labeled args that come after will override these via HashMap::insert.
                match v {
                    Value::Object(ref inner_rc) => {
                        for (k, fv) in inner_rc.borrow().fields.iter() {
                            vals.push(Value::Labeled { label: k.clone(), value: Box::new(fv.clone()) });
                        }
                    }
                    other => vals.push(other), // not a struct — pass through as-is
                }
            } else if let Some(label) = &arg.label {
                vals.push(Value::Labeled { label: label.clone(), value: Box::new(v) });
            } else {
                vals.push(v);
            }
        }
        Ok(vals)
    }

    pub(crate) fn call_value(&mut self, callee: Value, args: Vec<Value>, line: usize, in_throws_context: bool) -> Eval {
        match callee {
            Value::NativeFn { func, .. } => {
                func(&args, line).map_err(|sig| sig)
            }
            Value::Fn { decl, captured } => {
                self.call_fn(&decl, captured, args, line, in_throws_context)
            }
            Value::Closure { params, body, captured } => {
                self.call_closure(params, body, captured, args, line)
            }
            Value::Struct { decl, captured } => {
                // Struct is callable as constructor
                let obj = self.instantiate_struct_labeled(&decl, &captured, args, line)?;
                Ok(obj)
            }
            Value::EnumNamespace { name, variants, .. } => {
                // Calling enum namespace means accessing a variant constructor
                Err(err(format!("cannot call enum namespace '{}' directly; use .VariantName", name), line))
            }
            Value::EnumVariant { type_name, variant, .. } => {
                // Called as a constructor with args
                Ok(Value::EnumVariant { type_name, variant, fields: args })
            }
            Value::RustType { name } => {
                Self::construct_rust_type(&name, args, line)
            }
            other => Err(err(format!("'{}' is not callable", other.type_name()), line)),
        }
    }

    /// Construct a value from a Rust type name.
    /// Well-known collection types map to boring's native equivalents;
    /// everything else produces an opaque Object.
    pub(crate) fn construct_rust_type(name: &str, args: Vec<Value>, _line: usize) -> Eval {
        match name {
            // std::collections::HashMap  →  boring Dict
            "HashMap" => Ok(Value::Dict(vec![])),
            // std::collections::HashSet  →  boring Set
            "HashSet" => Ok(Value::Set(vec![])),
            // std::collections::BTreeMap →  boring Dict (ordered semantics at interp level)
            "BTreeMap" => Ok(Value::Dict(vec![])),
            // std::collections::BTreeSet →  boring Set
            "BTreeSet" => Ok(Value::Set(vec![])),
            // Vec  →  boring Array (optionally pre-filled)
            "Vec" | "VecDeque" => Ok(Value::Array(args)),
            // String  →  boring Str
            "String" => {
                match args.into_iter().next() {
                    Some(Value::Str(s)) => Ok(Value::Str(s)),
                    _ => Ok(Value::Str(String::new())),
                }
            }
            // Any other Rust type → opaque Object
            _ => Ok(make_object(name.to_string(), vec![])),
        }
    }

    pub(crate) fn item_pub_name<'a>(item: &'a Item) -> Option<(&'a str, bool)> {
        match item {
            Item::Fn(d)     => Some((&d.name, d.is_pub)),
            Item::Struct(d) => Some((&d.name, d.is_pub)),
            Item::Enum(d)   => Some((&d.name, d.is_pub)),
            Item::Let(d)    => Some((&d.name, d.is_pub)),
            Item::Mod(d)  => Some((&d.name, false)),
            Item::Use(_) | Item::Alias(_) | Item::Trait(_) | Item::Ext(_) | Item::Stmt(_) => None,
        }
    }

    pub(crate) fn exec_use_decl(&mut self, decl: &UseDecl, env: EnvRef) -> Result<(), Signal> {
        let path = &decl.path;
        if path.len() < 2 {
            return Ok(());
        }
        let prefix = path[0].as_str();
        let module = path[1].as_str();

        match (prefix, module) {
            // ── Rust stdlib (std::X) ─────────────────────────────────────────
            // use std.collections.HashMap, HashSet
            // use std.io.Write, BufRead
            // Registers each item as a RustType constructor.
            ("std", _rust_module) => {
                for item_name in &decl.items {
                    env.borrow_mut().define(
                        item_name,
                        Value::RustType { name: item_name.clone() },
                    );
                }
            }

            // ── External crates ──────────────────────────────────────────────
            // use crate.models.User, Post  — treat items as opaque RustTypes
            ("crate", _) => {
                for item_name in &decl.items {
                    env.borrow_mut().define(
                        item_name,
                        Value::RustType { name: item_name.clone() },
                    );
                }
            }

            _ => {
                // Not a stdlib module — fall back to the filesystem-based loader
                self.exec_use(decl, env)?;
            }
        }
        Ok(())
    }

}
