use super::*;
use std::collections::HashMap;
use std::rc::Rc;

impl Interpreter {
    pub(crate) fn call_method(&mut self, obj: Value, method: &str, args: Vec<Value>, line: usize, out_self: &mut Option<Value>) -> Eval {
        // KernelHandle<T> method dispatch — .wait() and .done()
        if let Value::KernelHandle { result } = &obj {
            return match method {
                "wait" => Ok(*result.clone()),
                "done" => Ok(Value::Bool(true)),
                _ => Err(err(format!("no method '{}' on KernelHandle", method), line)),
            };
        }
        // Future<T> method dispatch — .value() and .wait() as method-call syntax.
        // These mirror the field-access forms `future.value` / `future.wait`.
        if let Value::Future(inner) = &obj {
            match method {
                "value" => {
                    if !self.task_context {
                        return Err(err("'value()' requires a task context: the calling function must be marked 'task'", line));
                    }
                    return Ok(*inner.clone());
                }
                "wait" => {
                    if !self.task_context {
                        return Err(err("'wait()' requires a task context: the calling function must be marked 'task'", line));
                    }
                    return Ok(Value::Nil);
                }
                "done" => {
                    // Non-blocking poll — always true in the interpreter (eager evaluation)
                    return Ok(Value::Bool(true));
                }
                "cancel" => {
                    // Cancellation is not supported in the interpreter — no-op.
                    return Ok(Value::Nil);
                }
                _ => {}
            }
        }

        // RustType method dispatch — `HashMap.new()`, `Vec.with_capacity(n)`, etc.
        if let Value::RustType { ref name } = obj {
            // `new` → same as calling the constructor
            if method == "new" {
                return Self::construct_rust_type(name, args, line);
            }
            // Any other static method → produce an opaque Object result
            return Ok(make_object(name.clone(), vec![]));
        }

        // First check built-in methods for primitive types
        match &obj {
            Value::Str(_) => {
                if let Some(result) = self.call_str_method(&obj, method, &args, line)? {
                    return Ok(result);
                }
            }
            Value::Array(_) => {
                // pop/remove need special handling: they must set out_self to the
                // SHORTENED array AND return the extracted element (not the new array).
                if method == "pop" {
                    if let Value::Array(mut arr_owned) = obj.clone() {
                        if arr_owned.is_empty() {
                            return Err(err("pop: array is empty", line));
                        }
                        let last = arr_owned.pop().unwrap();
                        *out_self = Some(Value::Array(arr_owned));
                        return Ok(last);
                    }
                }
                if method == "remove" {
                    if let Value::Array(mut arr_owned) = obj.clone() {
                        let idx_val = args.first().cloned().unwrap_or(Value::Int(0));
                        let idx = self.expect_int(idx_val, line)?;
                        let idx = if idx < 0 { arr_owned.len() as i64 + idx } else { idx };
                        let removed = if idx >= 0 && (idx as usize) < arr_owned.len() {
                            arr_owned.remove(idx as usize)
                        } else {
                            Value::Nil
                        };
                        *out_self = Some(Value::Array(arr_owned));
                        return Ok(removed);
                    }
                }
                if let Some(result) = self.call_array_method(obj.clone(), method, args.clone(), line)? {
                    // For mutating array methods, set out_self so the caller can write back.
                    const MUTATING: &[&str] = &["push", "append", "insert", "sort", "reverse", "removeAt"];
                    if MUTATING.contains(&method) {
                        if let Value::Array(_) = &result {
                            *out_self = Some(result.clone());
                        }
                    }
                    return Ok(result);
                }
            }
            Value::Dict(_) => {
                if let Some(result) = self.call_dict_method(&obj, method, &args, line)? {
                    const MUTATING: &[&str] = &["set", "put", "remove", "removeAt"];
                    if MUTATING.contains(&method) {
                        if let Value::Dict(_) = &result {
                            *out_self = Some(result.clone());
                        }
                    }
                    return Ok(result);
                }
            }
            Value::Set(_) => {
                if let Some(result) = self.call_set_method(obj.clone(), method, args.clone(), line)? {
                    const MUTATING: &[&str] = &["add", "remove", "removeAt"];
                    if MUTATING.contains(&method) {
                        if let Value::Set(_) = &result {
                            *out_self = Some(result.clone());
                        }
                    }
                    return Ok(result);
                }
            }
            Value::Channel { buf, is_sender, .. } => {
                if method == "send" {
                    if !is_sender {
                        return Err(err("send called on a channel receiver", line));
                    }
                    let val = args.into_iter().next().unwrap_or(Value::Nil);
                    buf.borrow_mut().push_back(val);
                    return Ok(Value::Void);
                }
            }
            _ => {}
        }

        // `.upgrade()` — weak reference upgrade: returns the object if still live, else Nil.
        // In the interpreter, weak references share the same runtime value as strong refs;
        // the "gone-nil-on-drop" behaviour requires a full ownership system.
        if method == "upgrade" && args.is_empty() {
            return match &obj {
                Value::Object(_) | Value::EnumVariant { .. } => Ok(obj),
                _ => Ok(Value::Nil),
            };
        }

        // Struct method dispatch
        match obj.clone() {
            Value::Object(inner_rc) => {
                let (type_name, fields) = {
                    let inner = inner_rc.borrow();
                    (inner.type_name.clone(), inner.fields.clone())
                };
                let struct_val = self.global.borrow().get(&type_name);
                if let Some(Value::Struct { decl, captured }) = struct_val {
                    // Push type-param bindings inferred from this object's field values,
                    // plus any associated type definitions (e.g. `type Output = int`).
                    let pushed_type_params = !decl.type_params.is_empty() || !decl.assoc_type_defs.is_empty();
                    if pushed_type_params {
                        let mut bindings = if !decl.type_params.is_empty() {
                            Self::infer_struct_type_params(&decl, &fields)
                        } else {
                            HashMap::new()
                        };
                        for atd in &decl.assoc_type_defs {
                            bindings.insert(atd.name.clone(), atd.ty.clone());
                        }
                        self.type_param_stack.push(bindings);
                    }

                    let obj_clone = obj.clone();
                    let result = (|| -> Eval {
                        // Look in methods — use best-match overload resolution
                        if let Some(fn_decl) = Self::find_best_method(&decl.methods, method, &args).cloned() {
                            let fn_env = Env::child(Rc::clone(&captured));
                            fn_env.borrow_mut().define_mut("self", obj_clone.clone());
                            let result = self.call_fn(&fn_decl, Rc::clone(&fn_env), args, line, false)?;
                            *out_self = fn_env.borrow().get("self");
                            return Ok(result);
                        }
                        if method.is_empty() {
                            Err(err(format!("'{}' is not callable — no anonymous `def ()` or `req ()` defined", type_name), line))
                        } else {
                            Err(err(format!("no method '{}' on type '{}'", method, type_name), line))
                        }
                    })();

                    if pushed_type_params {
                        self.type_param_stack.pop();
                    }
                    return result;
                }
                Err(err(format!("no method '{}' on type '{}'", method, type_name), line))
            }
            Value::EnumNamespace { name, variants, .. } => {
                if let Some(variant_val) = variants.get(method) {
                    // Constructor call with args: Enum.Variant(fields)
                    if args.is_empty() {
                        Ok(variant_val.clone())
                    } else {
                        match variant_val.clone() {
                            Value::EnumVariant { type_name, variant, .. } =>
                                Ok(Value::EnumVariant { type_name, variant, fields: args }),
                            other => Ok(other),
                        }
                    }
                } else {
                    Err(err(format!("enum '{}' has no variant '{}'", name, method), line))
                }
            }
            Value::EnumVariant { ref type_name, .. } => {
                let ns = self.global.borrow().get(type_name);
                if let Some(Value::EnumNamespace { methods, captured, .. }) = ns {
                    let fn_decl = Self::find_best_method(&methods, method, &args).cloned();
                    if let Some(fn_decl) = fn_decl {
                        // Use the captured env from the enum's definition site, not self.global.
                        // This mirrors how struct methods work (Struct::captured) and allows
                        // enum methods to see module-level variables defined after the enum.
                        let fn_env = Env::child(captured);
                        fn_env.borrow_mut().define_mut("self", obj.clone());
                        let result = self.call_fn(&fn_decl, Rc::clone(&fn_env), args, line, false)?;
                        *out_self = fn_env.borrow().get("self");
                        return Ok(result);
                    }
                }
                Err(err(format!("method '{}' not found on enum variant", method), line))
            }
            Value::Tuple(ref elems) => {
                match method {
                    "length" | "count" if args.is_empty() => {
                        Ok(Value::Int(elems.len() as i64))
                    }
                    "isEmpty" if args.is_empty() => {
                        Ok(Value::Bool(elems.is_empty()))
                    }
                    "first" if args.is_empty() => {
                        elems.first().cloned().ok_or_else(|| {
                            err("'first()' called on empty tuple", line)
                        })
                    }
                    "last" if args.is_empty() => {
                        elems.last().cloned().ok_or_else(|| {
                            err("'last()' called on empty tuple", line)
                        })
                    }
                    "all" | "any" if args.len() == 1 => {
                        let is_all = method == "all";
                        let closure = args.into_iter().next().unwrap_or(Value::Nil);
                        let elems = elems.clone();
                        for item in elems {
                            let r = self.call_value(closure.clone(), vec![item], line, false)?;
                            let b = self.expect_bool(r, line)?;
                            if is_all && !b { return Ok(Value::Bool(false)); }
                            if !is_all && b  { return Ok(Value::Bool(true)); }
                        }
                        Ok(Value::Bool(is_all))
                    }
                    "map" if args.len() == 1 => {
                        // Apply the closure to each slot independently, returning a new tuple.
                        // Heterogeneous-safe: each element is passed individually.
                        let closure = args.into_iter().next().unwrap_or(Value::Nil);
                        let elems = elems.clone();
                        let mut result = Vec::with_capacity(elems.len());
                        for item in elems {
                            result.push(self.call_value(closure.clone(), vec![item], line, false)?);
                        }
                        Ok(Value::Tuple(result))
                    }
                    _ => Err(err(format!("no method '{}' on Tuple", method), line)),
                }
            }
            other => {
                Err(err(format!("no method '{}' on {}", method, other.type_name()), line))
            }
        }
    }

    /// Find the best-matching overload in `methods` for the given `method_name` and `args`.
    /// When there is only one candidate with that name, return it directly.
    /// With multiple candidates, pick the one whose parameter types all match the runtime args.
    fn find_best_method<'a>(methods: &'a [FnDecl], method_name: &str, args: &[Value]) -> Option<&'a FnDecl> {
        let candidates: Vec<&FnDecl> = methods.iter().filter(|m| m.name == method_name).collect();
        if candidates.len() <= 1 {
            return candidates.into_iter().next();
        }
        // Multiple overloads — find best arity + type match
        for decl in &candidates {
            let min_args = decl.params.iter().filter(|p| p.default.is_none()).count();
            let max_args = decl.params.len();
            if args.len() < min_args || args.len() > max_args {
                continue;
            }
            let all_match = decl.params.iter().zip(args.iter()).all(|(p, v)| {
                match &p.ty {
                    None => true,
                    Some(ty) => Self::value_matches_type_static(v, ty),
                }
            });
            if all_match {
                return Some(decl);
            }
        }
        // Fallback: return first candidate
        candidates.into_iter().next()
    }

    /// Simplified static type check for overload resolution (no self reference needed).
    fn value_matches_type_static(val: &Value, ty: &crate::ast::Type) -> bool {
        use crate::ast::Type;
        match ty {
            Type::Int    => matches!(val, Value::Int(_)),
            Type::Uint   => matches!(val, Value::Uint(_)),
            Type::Float  => matches!(val, Value::Float(_)),
            Type::Str    => matches!(val, Value::Str(_)),
            Type::Bool   => matches!(val, Value::Bool(_)),
            Type::Qualified(inner, _) => Self::value_matches_type_static(val, inner),
            Type::Named(name) => match name.as_str() {
                "int"    => matches!(val, Value::Int(_)),
                "uint"   => matches!(val, Value::Uint(_)),
                "float"  => matches!(val, Value::Float(_)),
                "bool"   => matches!(val, Value::Bool(_)),
                "string" => matches!(val, Value::Str(_)),
                _ => true, // Unknown named type — don't reject
            },
            Type::Optional(_) => true, // Accept any value for optional params
            Type::Array(_)    => matches!(val, Value::Array(_)),
            Type::Dict(_, _)  => matches!(val, Value::Dict(_)),
            Type::Set(_)      => matches!(val, Value::Set(_)),
            _ => true,
        }
    }

    pub(crate) fn call_str_method(&mut self, obj: &Value, method: &str, args: &[Value], line: usize) -> Result<Option<Value>, Signal> {
        let s = match obj {
            Value::Str(s) => s.clone(),
            _ => unreachable!(),
        };
        match method {
            "len" => Ok(Some(Value::Int(s.chars().count() as i64))),
            "contains" => {
                let sub = self.expect_str(args.get(0).cloned().unwrap_or(Value::Nil), line)?;
                Ok(Some(Value::Bool(s.contains(sub.as_str()))))
            }
            "startsWith" => {
                let sub = self.expect_str(args.get(0).cloned().unwrap_or(Value::Nil), line)?;
                Ok(Some(Value::Bool(s.starts_with(sub.as_str()))))
            }
            "endsWith" => {
                let sub = self.expect_str(args.get(0).cloned().unwrap_or(Value::Nil), line)?;
                Ok(Some(Value::Bool(s.ends_with(sub.as_str()))))
            }
            "split" => {
                let sep = self.expect_str(args.get(0).cloned().unwrap_or(Value::Nil), line)?;
                let parts: Vec<Value> = s.split(sep.as_str()).map(|p| Value::Str(p.to_string())).collect();
                Ok(Some(Value::Array(parts)))
            }
            "trim" => Ok(Some(Value::Str(s.trim().to_string()))),
            "trimStart" => Ok(Some(Value::Str(s.trim_start().to_string()))),
            "trimEnd"   => Ok(Some(Value::Str(s.trim_end().to_string()))),
            "upper" | "toUpper" | "toUpperCase" | "uppercased" => Ok(Some(Value::Str(s.to_uppercase()))),
            "lower" | "toLower" | "toLowerCase" | "lowercased" => Ok(Some(Value::Str(s.to_lowercase()))),
            "replace" => {
                let from = self.expect_str(args.get(0).cloned().unwrap_or(Value::Nil), line)?;
                let to = self.expect_str(args.get(1).cloned().unwrap_or(Value::Nil), line)?;
                Ok(Some(Value::Str(s.replace(from.as_str(), to.as_str()))))
            }
            "chars" => {
                let chars: Vec<Value> = s.chars().map(|c| Value::Str(c.to_string())).collect();
                Ok(Some(Value::Array(chars)))
            }
            "lines" => {
                let lines: Vec<Value> = s.lines().map(|l| Value::Str(l.to_string())).collect();
                Ok(Some(Value::Array(lines)))
            }
            "slice" => {
                let char_count = s.chars().count();
                let start_i = self.expect_int(args.get(0).cloned().unwrap_or(Value::Int(0)), line)?;
                let start = if start_i < 0 {
                    (char_count as i64 + start_i).max(0) as usize
                } else {
                    (start_i as usize).min(char_count)
                };
                let end = match args.get(1).cloned() {
                    Some(v) => {
                        let end_i = self.expect_int(v, line)?;
                        if end_i < 0 {
                            (char_count as i64 + end_i).max(0) as usize
                        } else {
                            (end_i as usize).min(char_count)
                        }
                    }
                    None => char_count,
                };
                let result: String = s.chars().skip(start).take(end.saturating_sub(start)).collect();
                Ok(Some(Value::Str(result)))
            }
            "repeat" => {
                let n = self.expect_int(args.get(0).cloned().unwrap_or(Value::Int(0)), line)? as usize;
                Ok(Some(Value::Str(s.repeat(n))))
            }
            "parseInt" => {
                match s.trim().parse::<i64>() {
                    Ok(n) => Ok(Some(Value::Int(n))),
                    Err(_) => Ok(Some(Value::Nil)),
                }
            }
            "parseFloat" => {
                match s.trim().parse::<f64>() {
                    Ok(f) => Ok(Some(Value::Float(f))),
                    Err(_) => Ok(Some(Value::Nil)),
                }
            }
            "indexOf" => {
                let sub = self.expect_str(args.get(0).cloned().unwrap_or(Value::Nil), line)?;
                match s.find(sub.as_str()) {
                    Some(i) => Ok(Some(Value::Int(i as i64))),
                    None => Ok(Some(Value::Nil)),
                }
            }
            "isEmpty" => Ok(Some(Value::Bool(s.is_empty()))),
            _ => Ok(None),
        }
    }

    pub(crate) fn call_array_method(&mut self, obj: Value, method: &str, args: Vec<Value>, line: usize) -> Result<Option<Value>, Signal> {
        let arr = match obj {
            Value::Array(a) => a,
            _ => unreachable!(),
        };
        match method {
            "len" | "length" => Ok(Some(Value::Int(arr.len() as i64))),
            "push" => {
                let mut new_arr = arr;
                new_arr.push(args.into_iter().next().unwrap_or(Value::Nil));
                Ok(Some(Value::Array(new_arr)))
            }
            "contains" => {
                let target = args.into_iter().next().unwrap_or(Value::Nil);
                Ok(Some(Value::Bool(arr.contains(&target))))
            }
            "first" => Ok(Some(arr.first().cloned().unwrap_or(Value::Nil))),
            "last" => Ok(Some(arr.last().cloned().unwrap_or(Value::Nil))),
            "reverse" => {
                let mut new_arr = arr;
                new_arr.reverse();
                Ok(Some(Value::Array(new_arr)))
            }
            "sort" => {
                let mut new_arr = arr;
                new_arr.sort_by(|a, b| {
                    match (a, b) {
                        (Value::Int(x), Value::Int(y)) => x.cmp(y),
                        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Str(x), Value::Str(y)) => x.cmp(y),
                        _ => std::cmp::Ordering::Equal,
                    }
                });
                Ok(Some(Value::Array(new_arr)))
            }
            "map" => {
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                let mut result = Vec::new();
                for item in arr {
                    let r = self.call_value(closure.clone(), vec![item], line, false)?;
                    result.push(r);
                }
                Ok(Some(Value::Array(result)))
            }
            "filter" => {
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                let mut result = Vec::new();
                for item in arr {
                    let r = self.call_value(closure.clone(), vec![item.clone()], line, false)?;
                    let b = self.expect_bool(r, line)?;
                    if b { result.push(item); }
                }
                Ok(Some(Value::Array(result)))
            }
            "reduce" => {
                // reduce(init, closure) — initial value first, closure second
                let init = args.get(0).cloned().unwrap_or(Value::Nil);
                let closure = args.get(1).cloned().unwrap_or(Value::Nil);
                let mut acc = init;
                for item in arr {
                    acc = self.call_value(closure.clone(), vec![acc, item], line, false)?;
                }
                Ok(Some(acc))
            }
            "join" => {
                let sep = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => String::new(),
                };
                let parts: Vec<String> = arr.iter().map(|v| format!("{}", v)).collect();
                Ok(Some(Value::Str(parts.join(&sep))))
            }
            "any" => {
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                for item in arr {
                    let r = self.call_value(closure.clone(), vec![item], line, false)?;
                    if self.expect_bool(r, line)? { return Ok(Some(Value::Bool(true))); }
                }
                Ok(Some(Value::Bool(false)))
            }
            "all" => {
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                for item in arr {
                    let r = self.call_value(closure.clone(), vec![item], line, false)?;
                    if !self.expect_bool(r, line)? { return Ok(Some(Value::Bool(false))); }
                }
                Ok(Some(Value::Bool(true)))
            }
            "find" => {
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                for item in arr {
                    let r = self.call_value(closure.clone(), vec![item.clone()], line, false)?;
                    if self.expect_bool(r, line)? { return Ok(Some(item)); }
                }
                Ok(Some(Value::Nil))
            }
            "indexOf" => {
                let target = args.into_iter().next().unwrap_or(Value::Nil);
                for (i, item) in arr.iter().enumerate() {
                    if item == &target { return Ok(Some(Value::Int(i as i64))); }
                }
                Ok(Some(Value::Nil))
            }
            "flatMap" => {
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                let mut result = Vec::new();
                for item in arr {
                    let r = self.call_value(closure.clone(), vec![item], line, false)?;
                    match r {
                        Value::Array(inner) => result.extend(inner),
                        other => result.push(other),
                    }
                }
                Ok(Some(Value::Array(result)))
            }
            "flat" => {
                let mut result = Vec::new();
                for item in arr {
                    match item {
                        Value::Array(inner) => result.extend(inner),
                        other => result.push(other),
                    }
                }
                Ok(Some(Value::Array(result)))
            }
            "zip" => {
                let other = match args.into_iter().next() {
                    Some(Value::Array(a)) => a,
                    _ => vec![],
                };
                let result: Vec<Value> = arr.into_iter().zip(other.into_iter())
                    .map(|(a, b)| Value::Tuple(vec![a, b]))
                    .collect();
                Ok(Some(Value::Array(result)))
            }
            "enumerate" => {
                let result: Vec<Value> = arr.into_iter().enumerate()
                    .map(|(i, v)| Value::Tuple(vec![Value::Int(i as i64), v]))
                    .collect();
                Ok(Some(Value::Array(result)))
            }
            "slice" => {
                let len = arr.len();
                let start_i = self.expect_int(args.get(0).cloned().unwrap_or(Value::Int(0)), line)?;
                let start = if start_i < 0 {
                    (len as i64 + start_i).max(0) as usize
                } else {
                    (start_i as usize).min(len)
                };
                let end = match args.get(1).cloned() {
                    Some(v) => {
                        let end_i = self.expect_int(v, line)?;
                        if end_i < 0 {
                            (len as i64 + end_i).max(0) as usize
                        } else {
                            (end_i as usize).min(len)
                        }
                    }
                    None => len,
                };
                Ok(Some(Value::Array(arr[start..end.max(start)].to_vec())))
            }
            "insert" => {
                let idx_i = self.expect_int(args.get(0).cloned().unwrap_or(Value::Int(0)), line)?;
                let val = args.get(1).cloned().unwrap_or(Value::Nil);
                let mut new_arr = arr;
                let idx = if idx_i < 0 {
                    (new_arr.len() as i64 + idx_i).max(0) as usize
                } else {
                    (idx_i as usize).min(new_arr.len())
                };
                new_arr.insert(idx, val);
                Ok(Some(Value::Array(new_arr)))
            }
            "remove" => {
                let idx = self.expect_int(args.get(0).cloned().unwrap_or(Value::Int(0)), line)?;
                let mut new_arr = arr;
                let idx = if idx < 0 { new_arr.len() as i64 + idx } else { idx };
                if idx >= 0 && (idx as usize) < new_arr.len() {
                    new_arr.remove(idx as usize);
                }
                Ok(Some(Value::Array(new_arr)))
            }
            "append" => {
                let other = match args.into_iter().next() {
                    Some(Value::Array(a)) => a,
                    _ => vec![],
                };
                let mut new_arr = arr;
                new_arr.extend(other);
                Ok(Some(Value::Array(new_arr)))
            }
            "count" => {
                match args.into_iter().next() {
                    Some(closure @ (Value::Closure { .. } | Value::Fn { .. } | Value::NativeFn { .. })) => {
                        let mut n = 0i64;
                        for item in arr {
                            let r = self.call_value(closure.clone(), vec![item], line, false)?;
                            if self.expect_bool(r, line)? { n += 1; }
                        }
                        Ok(Some(Value::Int(n)))
                    }
                    _ => Ok(Some(Value::Int(arr.len() as i64))),
                }
            }
            "min" => {
                let result = arr.iter().min_by(|a, b| match (a, b) {
                    (Value::Int(x), Value::Int(y)) => x.cmp(y),
                    (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                    _ => std::cmp::Ordering::Equal,
                }).cloned().unwrap_or(Value::Nil);
                Ok(Some(result))
            }
            "max" => {
                let result = arr.iter().max_by(|a, b| match (a, b) {
                    (Value::Int(x), Value::Int(y)) => x.cmp(y),
                    (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                    _ => std::cmp::Ordering::Equal,
                }).cloned().unwrap_or(Value::Nil);
                Ok(Some(result))
            }
            "sum" => {
                let mut int_sum = 0i64;
                let mut float_sum = 0.0f64;
                let mut has_float = false;
                for item in &arr {
                    match item {
                        Value::Int(n) => int_sum += n,
                        Value::Float(f) => { float_sum += f; has_float = true; }
                        _ => {}
                    }
                }
                if has_float {
                    Ok(Some(Value::Float(int_sum as f64 + float_sum)))
                } else {
                    Ok(Some(Value::Int(int_sum)))
                }
            }
            "isEmpty" => Ok(Some(Value::Bool(arr.is_empty()))),
            "reversed" => {
                let mut new_arr = arr;
                new_arr.reverse();
                Ok(Some(Value::Array(new_arr)))
            }
            "sorted" => {
                let mut new_arr = arr;
                new_arr.sort_by(|a, b| match (a, b) {
                    (Value::Int(x), Value::Int(y)) => x.cmp(y),
                    (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                    (Value::Str(x), Value::Str(y)) => x.cmp(y),
                    _ => std::cmp::Ordering::Equal,
                });
                Ok(Some(Value::Array(new_arr)))
            }
            "sortedBy" => {
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                let mut new_arr = arr;
                let mut sort_err: Option<Signal> = None;
                new_arr.sort_by(|a, b| {
                    if sort_err.is_some() { return std::cmp::Ordering::Equal; }
                    // Call closure(a, b) — returns negative/zero/positive or Bool
                    // We need &mut self but sort_by takes an Fn. Use a workaround:
                    // Return Equal for now; real impl needs unsafe or different approach.
                    // We use a pre-pass to build a key array instead.
                    let _ = (a, b, &closure);
                    std::cmp::Ordering::Equal
                });
                // Pre-pass approach: map each element to a sort key, then sort by key index
                let keys: Vec<Value> = {
                    let mut ks = Vec::new();
                    for item in &new_arr {
                        match self.call_value(closure.clone(), vec![item.clone()], line, false) {
                            Ok(k) => ks.push(k),
                            Err(e) => { sort_err = Some(e); break; }
                        }
                    }
                    ks
                };
                if let Some(e) = sort_err { return Err(e); }
                let mut indices: Vec<usize> = (0..new_arr.len()).collect();
                indices.sort_by(|&i, &j| {
                    match (&keys[i], &keys[j]) {
                        (Value::Int(x), Value::Int(y)) => x.cmp(y),
                        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Str(x), Value::Str(y)) => x.cmp(y),
                        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
                        _ => std::cmp::Ordering::Equal,
                    }
                });
                let sorted: Vec<Value> = indices.into_iter().map(|i| new_arr[i].clone()).collect();
                Ok(Some(Value::Array(sorted)))
            }
            "take" => {
                let n = self.expect_int(args.get(0).cloned().unwrap_or(Value::Int(0)), line)?;
                let n = (n.max(0)) as usize;
                Ok(Some(Value::Array(arr.into_iter().take(n).collect())))
            }
            "drop" => {
                let n = self.expect_int(args.get(0).cloned().unwrap_or(Value::Int(0)), line)?;
                let n = (n.max(0)) as usize;
                Ok(Some(Value::Array(arr.into_iter().skip(n).collect())))
            }
            "joined" => {
                let sep = match args.get(0) {
                    Some(Value::Str(s)) => s.clone(),
                    _ => String::new(),
                };
                let parts: Vec<String> = arr.iter().map(|v| format!("{}", v)).collect();
                Ok(Some(Value::Str(parts.join(&sep))))
            }
            // ── Index API ──────────────────────────────────────────────────────────
            "firstIndex" => {
                if arr.is_empty() {
                    Ok(Some(Value::Nil))
                } else {
                    Ok(Some(Value::Index(IndexValue::Array(0))))
                }
            }
            "nextIndex" => {
                match args.into_iter().next().unwrap_or(Value::Nil) {
                    Value::Index(IndexValue::Array(pos)) => {
                        let next = pos + 1;
                        if next < arr.len() {
                            Ok(Some(Value::Index(IndexValue::Array(next))))
                        } else {
                            Ok(Some(Value::Nil))
                        }
                    }
                    _ => Err(err("nextIndex: expected an ArrayIndex value", line)),
                }
            }
            "removeAt" => {
                match args.into_iter().next().unwrap_or(Value::Nil) {
                    Value::Index(IndexValue::Array(pos)) => {
                        if pos >= arr.len() {
                            return Err(err(format!("removeAt: index {} out of bounds (len {})", pos, arr.len()), line));
                        }
                        let mut new_arr = arr;
                        new_arr.remove(pos);
                        Ok(Some(Value::Array(new_arr)))
                    }
                    _ => Err(err("removeAt: expected an ArrayIndex value", line)),
                }
            }
            "getAt" => {
                match args.into_iter().next().unwrap_or(Value::Nil) {
                    Value::Index(IndexValue::Array(pos)) => {
                        arr.get(pos).cloned()
                            .map(Some)
                            .ok_or_else(|| err(format!("getAt: index {} out of bounds (len {})", pos, arr.len()), line))
                    }
                    _ => Err(err("getAt: expected an ArrayIndex value", line)),
                }
            }
            _ => Ok(None),
        }
    }

    pub(crate) fn call_dict_method(&mut self, obj: &Value, method: &str, args: &[Value], line: usize) -> Result<Option<Value>, Signal> {
        let pairs = match obj {
            Value::Dict(p) => p.clone(),
            _ => unreachable!(),
        };
        match method {
            "keys" => Ok(Some(Value::Array(pairs.into_iter().map(|(k, _)| k).collect()))),
            "values" => Ok(Some(Value::Array(pairs.into_iter().map(|(_, v)| v).collect()))),
            "len" => Ok(Some(Value::Int(pairs.len() as i64))),
            "contains" | "containsKey" | "has" => {
                let key = args.get(0).cloned().unwrap_or(Value::Nil);
                Ok(Some(Value::Bool(pairs.iter().any(|(k, _)| k == &key))))
            }
            "get" => {
                let key = args.get(0).cloned().unwrap_or(Value::Nil);
                let default = args.get(1).cloned().unwrap_or(Value::Nil);
                let found = pairs.into_iter().find(|(k, _)| k == &key).map(|(_, v)| v);
                Ok(Some(found.unwrap_or(default)))
            }
            "remove" => {
                let key = args.get(0).cloned().unwrap_or(Value::Nil);
                let new_pairs: Vec<(Value, Value)> = pairs.into_iter().filter(|(k, _)| k != &key).collect();
                Ok(Some(Value::Dict(new_pairs)))
            }
            "map" => {
                let closure = args.iter().next().cloned().unwrap_or(Value::Nil);
                let mut result = Vec::new();
                for (k, v) in pairs {
                    let new_v = self.call_value(closure.clone(), vec![k.clone(), v], line, false)?;
                    result.push((k, new_v));
                }
                Ok(Some(Value::Dict(result)))
            }
            "filter" => {
                let closure = args.iter().next().cloned().unwrap_or(Value::Nil);
                let mut result = Vec::new();
                for (k, v) in pairs {
                    let keep = self.call_value(closure.clone(), vec![k.clone(), v.clone()], line, false)?;
                    if self.expect_bool(keep, line)? { result.push((k, v)); }
                }
                Ok(Some(Value::Dict(result)))
            }
            "isEmpty" => Ok(Some(Value::Bool(pairs.is_empty()))),
            "count"   => Ok(Some(Value::Int(pairs.len() as i64))),
            "set" | "put" => {
                let key = args.get(0).cloned().unwrap_or(Value::Nil);
                let val = args.get(1).cloned().unwrap_or(Value::Nil);
                let mut new_pairs = pairs;
                let mut found = false;
                for (k, v) in &mut new_pairs {
                    if k == &key { *v = val.clone(); found = true; break; }
                }
                if !found { new_pairs.push((key, val)); }
                Ok(Some(Value::Dict(new_pairs)))
            }
            // ── Index API ──────────────────────────────────────────────────────────
            "firstIndex" => {
                if pairs.is_empty() {
                    Ok(Some(Value::Nil))
                } else {
                    Ok(Some(Value::Index(IndexValue::DictKey(Box::new(pairs[0].0.clone())))))
                }
            }
            "nextIndex" => {
                match args.iter().next().cloned().unwrap_or(Value::Nil) {
                    Value::Index(IndexValue::DictKey(key)) => {
                        let pos = pairs.iter().position(|(k, _)| k == &*key);
                        match pos {
                            Some(p) if p + 1 < pairs.len() => {
                                Ok(Some(Value::Index(IndexValue::DictKey(Box::new(pairs[p + 1].0.clone())))))
                            }
                            _ => Ok(Some(Value::Nil)),
                        }
                    }
                    _ => Err(err("nextIndex: expected a DictIndex value", line)),
                }
            }
            "removeAt" => {
                match args.iter().next().cloned().unwrap_or(Value::Nil) {
                    Value::Index(IndexValue::DictKey(key)) => {
                        let new_pairs: Vec<(Value, Value)> =
                            pairs.into_iter().filter(|(k, _)| k != &*key).collect();
                        Ok(Some(Value::Dict(new_pairs)))
                    }
                    _ => Err(err("removeAt: expected a DictIndex value", line)),
                }
            }
            _ => Ok(None),
        }
    }

    pub(crate) fn call_set_method(&mut self, obj: Value, method: &str, args: Vec<Value>, line: usize) -> Result<Option<Value>, Signal> {
        let set = match obj {
            Value::Set(s) => s,
            _ => unreachable!(),
        };
        match method {
            "add" => {
                let val = args.into_iter().next().unwrap_or(Value::Nil);
                let mut new_set = set;
                if !new_set.contains(&val) { new_set.push(val); }
                Ok(Some(Value::Set(new_set)))
            }
            "remove" => {
                let val = args.into_iter().next().unwrap_or(Value::Nil);
                let new_set: Vec<Value> = set.into_iter().filter(|v| v != &val).collect();
                Ok(Some(Value::Set(new_set)))
            }
            "contains" => {
                let val = args.into_iter().next().unwrap_or(Value::Nil);
                Ok(Some(Value::Bool(set.contains(&val))))
            }
            "toArray" => Ok(Some(Value::Array(set))),
            "isEmpty" => Ok(Some(Value::Bool(set.is_empty()))),
            "count" | "len" | "length" => Ok(Some(Value::Int(set.len() as i64))),
            "union" => {
                let other = match args.into_iter().next() {
                    Some(Value::Set(s)) => s,
                    Some(Value::Array(a)) => a,
                    _ => vec![],
                };
                let mut new_set = set;
                for v in other {
                    if !new_set.contains(&v) { new_set.push(v); }
                }
                Ok(Some(Value::Set(new_set)))
            }
            "intersection" => {
                let other = match args.into_iter().next() {
                    Some(Value::Set(s)) => s,
                    Some(Value::Array(a)) => a,
                    _ => vec![],
                };
                let new_set: Vec<Value> = set.into_iter().filter(|v| other.contains(v)).collect();
                Ok(Some(Value::Set(new_set)))
            }
            "difference" => {
                let other = match args.into_iter().next() {
                    Some(Value::Set(s)) => s,
                    Some(Value::Array(a)) => a,
                    _ => vec![],
                };
                let new_set: Vec<Value> = set.into_iter().filter(|v| !other.contains(v)).collect();
                Ok(Some(Value::Set(new_set)))
            }
            "isSubset" => {
                let other = match args.into_iter().next() {
                    Some(Value::Set(s)) => s,
                    Some(Value::Array(a)) => a,
                    _ => vec![],
                };
                Ok(Some(Value::Bool(set.iter().all(|v| other.contains(v)))))
            }
            // ── Index API ──────────────────────────────────────────────────────────
            // Set indices are READ-ONLY — writing through them is forbidden because
            // modifying an element would break the uniqueness invariant.
            "firstIndex" => {
                if set.is_empty() {
                    Ok(Some(Value::Nil))
                } else {
                    Ok(Some(Value::Index(IndexValue::Set(0))))
                }
            }
            "nextIndex" => {
                match args.into_iter().next().unwrap_or(Value::Nil) {
                    Value::Index(IndexValue::Set(pos)) => {
                        let next = pos + 1;
                        if next < set.len() {
                            Ok(Some(Value::Index(IndexValue::Set(next))))
                        } else {
                            Ok(Some(Value::Nil))
                        }
                    }
                    _ => Err(err("nextIndex: expected a SetIndex value", line)),
                }
            }
            // removeAt on a set: remove the element at this opaque position.
            // Returns the new set (mutation via write-back, listed in MUTATING).
            "removeAt" => {
                match args.into_iter().next().unwrap_or(Value::Nil) {
                    Value::Index(IndexValue::Set(pos)) => {
                        if pos >= set.len() {
                            return Err(err(format!("removeAt: set index {} out of bounds (len {})", pos, set.len()), line));
                        }
                        let mut new_set = set;
                        new_set.remove(pos);
                        Ok(Some(Value::Set(new_set)))
                    }
                    _ => Err(err("removeAt: expected a SetIndex value", line)),
                }
            }
            // getAt — explicit read via SetIndex; useful in transpiled code where
            // `set[i]` is not valid Rust syntax for HashSet.
            "getAt" => {
                match args.into_iter().next().unwrap_or(Value::Nil) {
                    Value::Index(IndexValue::Set(pos)) => {
                        set.get(pos).cloned()
                            .map(Some)
                            .ok_or_else(|| err(format!("getAt: index {} out of bounds (len {})", pos, set.len()), line))
                    }
                    _ => Err(err("getAt: expected a SetIndex value", line)),
                }
            }
            _ => Ok(None),
        }
    }

    pub(crate) fn get_field(&mut self, obj: Value, field: &str, line: usize) -> Eval {
        match obj {
            // KernelHandle.done() / .wait — simulation: kernel already ran, always done.
            Value::KernelHandle { result } if field == "done" => {
                let _ = result; // mark as used
                Ok(Value::Bool(true))
            }
            Value::KernelHandle { result } if field == "wait" => Ok(*result),
            // Future.done() — always true in the interpreter (futures are evaluated eagerly)
            Value::Future(_) if field == "done" => Ok(Value::Bool(true)),
            Value::Future(inner) if field == "value" => {
                if !self.task_context {
                    return Err(err("'.value' requires a task context: the calling function must be marked 'task'", line));
                }
                Ok(*inner)
            }
            Value::Future(_) if field == "wait" => {
                // wait: join the future and discard the result (void)
                if !self.task_context {
                    return Err(err("'.wait' requires a task context: the calling function must be marked 'task'", line));
                }
                Ok(Value::Nil)
            }
            Value::Object(ref inner_rc) => {
                let (type_name, fields) = {
                    let inner = inner_rc.borrow();
                    (inner.type_name.clone(), inner.fields.clone())
                };
                // Check for zero-param req method acting as property getter
                let struct_val = self.global.borrow().get(&type_name);
                if let Some(Value::Struct { decl, captured }) = struct_val {
                    if let Some(method) = decl.methods.iter().find(|m| m.name == field && !m.mutating && m.params.is_empty()).cloned() {
                        if method.task && !self.task_context {
                            return Err(err(
                                format!("req method '{}' is a task and must be called from a task context", method.name),
                                line,
                            ));
                        }
                        let pushed = !decl.type_params.is_empty();
                        if pushed {
                            let bindings = Self::infer_struct_type_params(&decl, &fields);
                            self.type_param_stack.push(bindings);
                        }
                        let fn_env = Env::child(Rc::clone(&captured));
                        fn_env.borrow_mut().define_mut("self", obj.clone());
                        // Push a fresh defer frame so that any `defer:` inside the getter
                        // runs when the getter returns, not when the enclosing function exits.
                        self.defer_stack.push(Vec::new());
                        let result = self.eval_block_as_expr(&method.body, Rc::clone(&fn_env));
                        if let Some(frame) = self.defer_stack.pop() {
                            for deferred in frame.into_iter().rev() {
                                let _ = self.exec_block(&deferred, Rc::clone(&fn_env));
                            }
                        }
                        if pushed { self.type_param_stack.pop(); }
                        return result;
                    }
                }
                // Fall through to raw field access
                for (name, val) in &fields {
                    if name == field { return Ok(val.clone()); }
                }
                Err(err(format!("field '{}' not found on '{}'", field, type_name), line))
            }
            Value::EnumNamespace { name, variants, .. } => {
                if let Some(v) = variants.get(field) {
                    Ok(v.clone())
                } else {
                    Err(err(format!("enum '{}' has no variant '{}'", name, field), line))
                }
            }
            Value::EnumVariant { ref type_name, ref variant, ref fields } => {
                // Check named fields first using the stored EnumDecl
                let type_name_owned = type_name.clone();
                let variant_owned = variant.clone();
                let fields_owned = fields.clone();
                if let Some(enum_decl) = self.enums.get(&type_name_owned).cloned() {
                    if let Some(variant_decl) = enum_decl.variants.iter().find(|v| v.name == variant_owned) {
                        for (i, vf) in variant_decl.fields.iter().enumerate() {
                            if vf.name.as_deref() == Some(field) {
                                if let Some(val) = fields_owned.get(i) {
                                    return Ok(val.clone());
                                }
                            }
                        }
                    }
                }
                let ns = self.global.borrow().get(&type_name_owned);
                if let Some(Value::EnumNamespace { methods, .. }) = ns {
                    let method = methods.iter()
                        .find(|m| m.name == field && !m.mutating && m.params.is_empty())
                        .cloned();
                    if let Some(method) = method {
                        if method.task && !self.task_context {
                            return Err(err(
                                format!("req method '{}' is a task and must be called from a task context", method.name),
                                line,
                            ));
                        }
                        let fn_env = Env::child(Rc::clone(&self.global));
                        fn_env.borrow_mut().define_mut("self", obj.clone());
                        // Same defer-frame isolation as the struct getter above.
                        self.defer_stack.push(Vec::new());
                        let result = self.eval_block_as_expr(&method.body, Rc::clone(&fn_env));
                        if let Some(frame) = self.defer_stack.pop() {
                            for deferred in frame.into_iter().rev() {
                                let _ = self.exec_block(&deferred, Rc::clone(&fn_env));
                            }
                        }
                        return result;
                    }
                }
                Err(err(format!("enum variant has no field '{}'", field), line))
            }
            Value::Tuple(elems) => {
                // Tuple field access by index name: .0, .1, ...
                if let Ok(idx) = field.parse::<usize>() {
                    elems.get(idx).cloned().ok_or_else(|| {
                        err(format!("tuple index {} out of bounds", idx), line)
                    })
                } else {
                    Err(err(format!("invalid tuple field '{}'", field), line))
                }
            }
            Value::RustType { ref name } => {
                // Allow `TypeName.new` — returns itself as a constructor stub.
                // `HashMap.new()` will call construct_rust_type when invoked.
                if field == "new" {
                    Ok(Value::RustType { name: name.clone() })
                } else {
                    // Sub-type access: `HashMap.Entry` etc. — treat as opaque RustType
                    Ok(Value::RustType { name: format!("{}::{}", name, field) })
                }
            }
            Value::Array(ref arr) if field == "length" || field == "count" || field == "len" => {
                Ok(Value::Int(arr.len() as i64))
            }
            Value::Set(ref elems) if field == "length" || field == "count" || field == "len" => {
                Ok(Value::Int(elems.len() as i64))
            }
            Value::Dict(ref pairs) if field == "length" || field == "count" || field == "len" => {
                Ok(Value::Int(pairs.len() as i64))
            }
            Value::Str(ref s) if field == "length" || field == "len" => {
                Ok(Value::Int(s.chars().count() as i64))
            }
            other => Err(err(format!("cannot access field '{}' on {}", field, other.type_name()), line)),
        }
    }

    pub(crate) fn get_index(&mut self, obj: Value, idx: Value, line: usize) -> Eval {
        match obj {
            Value::Array(arr) => {
                let pos: usize = match idx {
                    Value::Index(IndexValue::Array(p)) => p,
                    other => {
                        let i = self.expect_int(other, line)?;
                        let i = if i < 0 { arr.len() as i64 + i } else { i };
                        if i < 0 || i as usize >= arr.len() {
                            return Err(err(format!("array index {} out of bounds (len {})", i, arr.len()), line));
                        }
                        i as usize
                    }
                };
                arr.get(pos).cloned()
                    .ok_or_else(|| err(format!("array index {} out of bounds (len {})", pos, arr.len()), line))
            }
            Value::Dict(pairs) => {
                // Accept either a DictIndex or a raw key value.
                let key = match idx {
                    Value::Index(IndexValue::DictKey(k)) => *k,
                    other => other,
                };
                for (k, v) in &pairs {
                    if k == &key { return Ok(v.clone()); }
                }
                Ok(Value::Nil) // Dict access returns nil for missing keys
            }
            Value::Set(set) => {
                // Set subscript is READ-ONLY and requires an opaque SetIndex.
                match idx {
                    Value::Index(IndexValue::Set(pos)) => {
                        set.get(pos).cloned()
                            .ok_or_else(|| err(format!("set index {} out of bounds (len {})", pos, set.len()), line))
                    }
                    _ => Err(err("set subscript requires a set index — use firstIndex() / nextIndex()", line)),
                }
            }
            Value::Str(s) => {
                let i = self.expect_int(idx, line)?;
                let chars: Vec<char> = s.chars().collect();
                let i = if i < 0 { chars.len() as i64 + i } else { i };
                if i < 0 || i as usize >= chars.len() {
                    Err(err(format!("string index {} out of bounds", i), line))
                } else {
                    Ok(Value::Str(chars[i as usize].to_string()))
                }
            }
            other => Err(err(format!("cannot index into {}", other.type_name()), line)),
        }
    }

    pub(crate) fn assign(&mut self, target: &Expr, val: Value, env: EnvRef, line: usize) -> Result<(), Signal> {
        match &target.kind {
            ExprKind::Var(name) => {
                // `_` is the explicit discard sink — evaluate for side effects, store nothing.
                if name == "_" { return Ok(()); }
                // Use a scoped block so the borrow_mut() is dropped before any recursive call.
                let set_result = env.borrow_mut().set(name, val.clone());
                match set_result {
                    Ok(true) => {}
                    Ok(false) => {
                        // Not found in scope — check implicit self before creating a new local.
                        let has_self_field = env.borrow().get("self")
                            .map(|sv| matches!(&sv, Value::Object(rc) if rc.borrow().fields.iter().any(|(k,_)| k == name)))
                            .unwrap_or(false);
                        if has_self_field {
                            // Delegate to field assignment (handles mutability + setters).
                            let self_expr = Expr { kind: ExprKind::Var("self".to_string()), line };
                            let field_expr = Expr { kind: ExprKind::Field(Box::new(self_expr), name.clone()), line };
                            return self.assign(&field_expr, val, Rc::clone(&env), line);
                        }
                        // Genuinely new local — define as mutable (bare assignment acts as declaration)
                        env.borrow_mut().define_mut(name, val);
                    }
                    Err(()) => return Err(err(format!("cannot assign to immutable variable '{}'", name), line)),
                }
                Ok(())
            }
            ExprKind::Field(obj_expr, field) => {
                // Type-level assignment: `Counter.count = v`
                if let ExprKind::Var(type_name) = &obj_expr.kind {
                    if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        let is_struct = self.global.borrow().get(type_name)
                            .map(|v| matches!(v, Value::Struct { .. }))
                            .unwrap_or(false);
                        if is_struct {
                            let key = format!("{}::{}", type_name, field);
                            // Check mutability: type let is immutable
                            let is_mutable = {
                                let g = self.global.borrow();
                                if let Some(Value::Struct { ref decl, .. }) = g.get(type_name) {
                                    decl.type_vars.iter()
                                        .find(|tv| tv.name == *field)
                                        .map(|tv| tv.mutable)
                                        .unwrap_or(false)
                                } else { false }
                            };
                            if !is_mutable {
                                return Err(err(
                                    format!("cannot assign to 'type let {}' — use 'type var' for mutable type variables", field),
                                    line,
                                ));
                            }
                            // Check if type set exists for this field (skip when already inside a setter)
                            let setter_opt = if !self.in_type_setter {
                                let g = self.global.borrow();
                                if let Some(Value::Struct { ref decl, .. }) = g.get(type_name) {
                                    decl.type_methods.iter()
                                        .find(|m| m.kind == crate::ast::TypeMethodKind::Set && m.name == *field)
                                        .cloned()
                                } else { None }
                            } else { None };
                            if let Some(setter) = setter_opt {
                                let captured = {
                                    let g = self.global.borrow();
                                    if let Some(Value::Struct { captured, .. }) = g.get(type_name) {
                                        Rc::clone(&captured)
                                    } else { Rc::clone(&env) }
                                };
                                self.in_type_setter = true;
                                let r = self.call_type_method(type_name, &setter, vec![val], captured, line).map(|_| ());
                                self.in_type_setter = false;
                                return r;
                            }
                            self.type_var_store.insert(key, val);
                            return Ok(());
                        }
                    }
                }
                // Check: cannot mutate a let binding's field
                if let ExprKind::Var(binding_name) = &obj_expr.kind {
                    if !env.borrow().is_mutable(binding_name) {
                        return Err(err(
                            format!("cannot mutate let binding '{}'", binding_name),
                            line,
                        ));
                    }
                }
                let obj = self.eval_expr(obj_expr, Rc::clone(&env))?;
                match obj {
                    Value::Object(ref inner_rc) => {
                        let (type_name, fields) = {
                            let inner = inner_rc.borrow();
                            (inner.type_name.clone(), inner.fields.clone())
                        };
                        // Check for setter first
                        let struct_val = self.global.borrow().get(&type_name);
                        if let Some(Value::Struct { decl, captured }) = struct_val {
                            if let Some(setter) = decl.setters.iter().find(|s| s.name == *field).cloned() {
                                if setter.task && !self.task_context {
                                    return Err(err(
                                        format!("setter '{}' is a task and must be called from a task context", setter.name),
                                        line,
                                    ));
                                }
                                let pushed = !decl.type_params.is_empty();
                                if pushed {
                                    let bindings = Self::infer_struct_type_params(&decl, &fields);
                                    self.type_param_stack.push(bindings);
                                }
                                let fn_env = Env::child(Rc::clone(&captured));
                                fn_env.borrow_mut().define_mut("self", obj.clone());
                                fn_env.borrow_mut().define(&setter.param_name, val);
                                let result = self.exec_block(&setter.body, Rc::clone(&fn_env));
                                if pushed { self.type_param_stack.pop(); }
                                result?;
                                // Write back self (setter may have mutated it)
                                if let Some(new_self) = fn_env.borrow().get("self") {
                                    self.assign(obj_expr, new_self, env, line)?;
                                }
                                return Ok(());
                            }
                        }
                        // Fall through to raw field write — check mutability first
                        // Do struct lookup BEFORE borrow_mut to avoid double borrow
                        {
                            let struct_val = self.global.borrow().get(&type_name);
                            if let Some(Value::Struct { decl, .. }) = struct_val {
                                if let Some(fd) = decl.fields.iter().find(|f| f.name == *field) {
                                    if !fd.mutable && !self.in_init_body {
                                        return Err(err(format!("cannot assign to immutable field '{}'", field), line));
                                    }
                                    // In a req (non-mutating) method, only transient fields may be written
                                    // (init bodies are always allowed to write any field)
                                    if !self.current_method_mutating && !fd.transient && !self.in_init_body {
                                        return Err(err(
                                            format!("cannot mutate non-transient field '{}' from a req method", field),
                                            line,
                                        ));
                                    }
                                }
                            }
                        }
                        // Mutate in-place through the Rc<RefCell<>> — no write-back needed
                        let mut inner_mut = inner_rc.borrow_mut();
                        for (k, v) in &mut inner_mut.fields {
                            if k == field { *v = val.clone(); break; }
                        }
                        return Ok(());
                    }
                    Value::EnumVariant { ref type_name, .. } => {
                        let ns = self.global.borrow().get(type_name);
                        if let Some(Value::EnumNamespace { setters, .. }) = ns {
                            if let Some(setter) = setters.iter().find(|s| s.name == *field).cloned() {
                                if setter.task && !self.task_context {
                                    return Err(err(
                                        format!("setter '{}' is a task and must be called from a task context", setter.name),
                                        line,
                                    ));
                                }
                                let fn_env = Env::child(Rc::clone(&self.global));
                                fn_env.borrow_mut().define_mut("self", obj.clone());
                                fn_env.borrow_mut().define(&setter.param_name, val);
                                self.exec_block(&setter.body, Rc::clone(&fn_env))?;
                                // Write back self (setter may have mutated it)
                                if let Some(new_self) = fn_env.borrow().get("self") {
                                    self.assign(obj_expr, new_self, env, line)?;
                                }
                                return Ok(());
                            }
                        }
                        return Err(err(format!("enum variant has no settable field '{}'", field), line));
                    }
                    _ => return Err(err("cannot assign field on non-object", line)),
                }
            }
            ExprKind::Index(obj_expr, idx_expr) => {
                let idx = self.eval_expr(idx_expr, Rc::clone(&env))?;
                let obj = self.eval_expr(obj_expr, Rc::clone(&env))?;
                match obj {
                    Value::Array(mut arr) => {
                        let pos: usize = match idx {
                            Value::Index(IndexValue::Array(p)) => p,
                            other => {
                                let i = self.expect_int(other, line)?;
                                let i = if i < 0 { arr.len() as i64 + i } else { i };
                                if i < 0 || i as usize >= arr.len() {
                                    return Err(err(format!("array index {} out of bounds (len {})", i, arr.len()), line));
                                }
                                i as usize
                            }
                        };
                        if pos < arr.len() { arr[pos] = val; }
                        self.assign(obj_expr, Value::Array(arr), env, line)?;
                    }
                    Value::Dict(mut pairs) => {
                        let key = match idx {
                            Value::Index(IndexValue::DictKey(k)) => *k,
                            other => other,
                        };
                        let mut found = false;
                        for (k, v) in &mut pairs {
                            if k == &key { *v = val.clone(); found = true; break; }
                        }
                        if !found { pairs.push((key, val)); }
                        self.assign(obj_expr, Value::Dict(pairs), env, line)?;
                    }
                    Value::Set(_) => {
                        return Err(err(
                            "cannot assign to a set element via index — \
                             set elements are their own keys; use remove() + add() instead",
                            line,
                        ));
                    }
                    _ => return Err(err("cannot index-assign on non-array/dict", line)),
                }
                Ok(())
            }
            _ => Err(err("invalid assignment target", line)),
        }
    }

    /// Convert a value to its display string, honouring `as string:` conversions.
    /// Falls back to `format!("{}", val)` when no conversion is defined.
    pub(crate) fn display_value(&mut self, val: Value, line: usize) -> Result<String, Signal> {
        if matches!(&val, Value::Object(_) | Value::EnumVariant { .. }) {
            let str_ty = Type::Str;
            match self.cast_value(val.clone(), &str_ty, line) {
                Ok(Value::Str(s)) => return Ok(s),
                _ => {}
            }
        }
        Ok(format!("{}", val))
    }

    /// Handle `print`, `write`, and log-level builtins with `as string:` conversions.
    pub(crate) fn call_display_builtin(&mut self, name: &str, args: &[Value], line: usize) -> Eval {
        let output = if args.len() >= 2 {
            if let Value::Str(fmt) = &args[0] {
                // Format string mode: `print "{}", val` — same logic as macro_format
                // but using display_value for each positional argument.
                let fmt = fmt.clone();
                let mut result = String::new();
                let mut arg_iter = args.iter().skip(1);
                let mut chars = fmt.chars().peekable();
                while let Some(c) = chars.next() {
                    if c == '{' {
                        match chars.peek() {
                            Some('}') => {
                                chars.next();
                                let val = arg_iter.next().cloned().unwrap_or(Value::Nil);
                                result.push_str(&self.display_value(val, line)?);
                            }
                            Some(':') => {
                                let mut spec = String::new();
                                chars.next();
                                for ch in chars.by_ref() {
                                    if ch == '}' { break; }
                                    spec.push(ch);
                                }
                                let val = arg_iter.next().unwrap_or(&Value::Nil);
                                let formatted = match spec.as_str() {
                                    "?" | "#?" => format!("{:?}", val),
                                    "x"  => if let Value::Int(n) = val { format!("{:x}", n) } else { format!("{}", val) },
                                    "X"  => if let Value::Int(n) = val { format!("{:X}", n) } else { format!("{}", val) },
                                    "b"  => if let Value::Int(n) = val { format!("{:b}", n) } else { format!("{}", val) },
                                    "o"  => if let Value::Int(n) = val { format!("{:o}", n) } else { format!("{}", val) },
                                    "e"  => if let Value::Float(f) = val { format!("{:e}", f) } else { format!("{}", val) },
                                    "E"  => if let Value::Float(f) = val { format!("{:E}", f) } else { format!("{}", val) },
                                    _ => format!("{}", val),
                                };
                                result.push_str(&formatted);
                            }
                            Some('{') => {
                                chars.next();
                                result.push('{');
                            }
                            _ => result.push(c),
                        }
                    } else if c == '}' && chars.peek() == Some(&'}') {
                        chars.next();
                        result.push('}');
                    } else {
                        result.push(c);
                    }
                }
                result
            } else {
                // Multiple non-format-string args: space-separated
                let parts: Vec<String> = args.iter()
                    .map(|v| self.display_value(v.clone(), line))
                    .collect::<Result<_, _>>()?;
                parts.join(" ")
            }
        } else {
            // Single arg (or zero)
            let parts: Vec<String> = args.iter()
                .map(|v| self.display_value(v.clone(), line))
                .collect::<Result<_, _>>()?;
            parts.join(" ")
        };

        match name {
            "print"  => println!("{}", output),
            "write"  => print!("{}", output),
            "error"  => eprintln!("[ERROR] {}", output),
            "warn"   => eprintln!("[WARN] {}", output),
            "info"   => eprintln!("[INFO] {}", output),
            "debug"  => eprintln!("[DEBUG] {}", output),
            "trace"  => eprintln!("[TRACE] {}", output),
            _        => println!("{}", output),
        }
        Ok(Value::Nil)
    }

    pub(crate) fn cast_value(&mut self, val: Value, ty: &Type, line: usize) -> Eval {
        // Resolve aliases (int→Int, string→Str, etc.) before dispatching
        let resolved = self.resolve_type(ty);
        let ty = &resolved;

        // Check for an as_decl on the object's struct first — user-defined conversions
        // take priority over built-in casts.
        if let Value::Object(ref inner_rc) = val {
            let type_name = inner_rc.borrow().type_name.clone();
            let struct_val = self.global.borrow().get(&type_name);
            if let Some(Value::Struct { decl, captured }) = struct_val {
                if let Some(as_decl) = decl.conversions.iter().find(|a| {
                    type_matches(&strip_qualifiers(&self.resolve_type(&a.ty)), strip_qualifiers(ty))
                }) {
                    let body = as_decl.body.clone();
                    let fn_env = Env::child(captured);
                    fn_env.borrow_mut().define_mut("self", val);
                    return self.eval_block_as_expr(&body, fn_env);
                }
            }
        }

        // Check for an as_decl on the enum variant's namespace.
        if let Value::EnumVariant { ref type_name, .. } = val {
            let ns = self.global.borrow().get(type_name);
            if let Some(Value::EnumNamespace { conversions, captured, .. }) = ns {
                if let Some(as_decl) = conversions.iter().find(|a| {
                    type_matches(&strip_qualifiers(&self.resolve_type(&a.ty)), strip_qualifiers(ty))
                }) {
                    let body = as_decl.body.clone();
                    let fn_env = Env::child(captured);
                    fn_env.borrow_mut().define_mut("self", val);
                    return self.eval_block_as_expr(&body, fn_env);
                }
            }
        }

        match ty {
            // Strip ownership qualifiers — `int` = Qualified(Int, Copy), etc.
            Type::Qualified(inner, _) => return self.cast_value(val, inner, line),
            Type::Int => match val {
                Value::Int(n) => Ok(Value::Int(n)),
                Value::Uint(n) => Ok(Value::Int(n as i64)),
                Value::Float(f) => Ok(Value::Int(f as i64)),
                Value::Str(s) => Ok(s.trim().parse::<i64>().map(Value::Int).unwrap_or(Value::Nil)),
                Value::Bool(b) => Ok(Value::Int(if b { 1 } else { 0 })),
                _ => Ok(Value::Nil),
            },
            Type::Uint => match val {
                Value::Uint(n) => Ok(Value::Uint(n)),
                Value::Int(n) if n >= 0 => Ok(Value::Uint(n as u64)),
                Value::Float(f) if f >= 0.0 => Ok(Value::Uint(f as u64)),
                Value::Str(s) => Ok(s.trim().parse::<u64>().map(Value::Uint).unwrap_or(Value::Nil)),
                _ => Ok(Value::Nil),
            },
            Type::Float => match val {
                Value::Float(f) => Ok(Value::Float(f)),
                Value::Int(n) => Ok(Value::Float(n as f64)),
                Value::Uint(n) => Ok(Value::Float(n as f64)),
                Value::Str(s) => Ok(s.trim().parse::<f64>().map(Value::Float).unwrap_or(Value::Nil)),
                _ => Ok(Value::Nil),
            },
            Type::Str => Ok(Value::Str(format!("{}", val))),
            Type::Bool => match val {
                Value::Bool(b) => Ok(Value::Bool(b)),
                _ => Ok(Value::Nil),
            },
            // `x as T?` — cast to the inner type; Nil is returned on failure.
            Type::Optional(inner) => self.cast_value(val, inner, line),
            _ => Ok(val),
        }
    }

    pub(crate) fn match_pattern(&self, pattern: &Pattern, value: &Value, bindings: &mut HashMap<String, Value>) -> bool {
        match pattern {
            Pattern::Wildcard => true,
            Pattern::Bind(name) => {
                bindings.insert(name.clone(), value.clone());
                true
            }
            Pattern::Lit(lit) => match (lit, value) {
                (LitPattern::Int(n), Value::Int(v)) => n == v,
                // For NaN: treat NaN == NaN as true (pattern matching, not arithmetic).
                (LitPattern::Float(f), Value::Float(v)) => (f.is_nan() && v.is_nan()) || f == v,
                (LitPattern::Str(s), Value::Str(v)) => s == v,
                (LitPattern::Bool(b), Value::Bool(v)) => b == v,
                (LitPattern::Nil, Value::Nil) => true,
                _ => false,
            },
            Pattern::None => matches!(value, Value::Nil),
            Pattern::Some(inner) => {
                match value {
                    // User-defined enum variant named "Some"/"some" — treat like
                    // a variant match: bind the inner pattern to the payload field(s).
                    Value::EnumVariant { variant, fields, .. }
                        if variant == "Some" || variant == "some" =>
                    {
                        if fields.len() == 1 {
                            self.match_pattern(inner, &fields[0], bindings)
                        } else if fields.is_empty() {
                            // Bare variant Some (no payload) — wildcard always matches
                            matches!(inner.as_ref(), Pattern::Wildcard)
                        } else {
                            false
                        }
                    }
                    // Boring Optional: nil does not match Some
                    Value::Nil => false,
                    // Any other non-nil value: unwrap into inner pattern
                    _ => self.match_pattern(inner, value, bindings),
                }
            }
            Pattern::Variant(name, sub_pats) => {
                // `Some(x)` / `None` parses as Variant("Some"/"None", ...) — treat as an
                // Option unwrap ONLY when the value is not a real enum variant named "Some"/"None".
                // If it *is* a real enum variant with that name, fall through to normal matching.
                let value_is_enum_with_name = matches!(value,
                    Value::EnumVariant { variant: v, .. } if v == name.as_str()
                );
                if !value_is_enum_with_name {
                    if (name == "Some" || name == "some") && sub_pats.len() == 1 {
                        return if matches!(value, Value::Nil) {
                            false
                        } else {
                            self.match_pattern(&sub_pats[0], value, bindings)
                        };
                    }
                    if (name == "None" || name == "none") && sub_pats.is_empty() {
                        return matches!(value, Value::Nil);
                    }
                }
                match value {
                    Value::EnumVariant { type_name, variant, fields, .. } => {
                        // Qualified pattern `TypeName::Variant` (from `TypeName.Variant` in source)
                        // or bare `Variant` — both must match the enum variant.
                        let name_matches = if let Some((enum_ty, var_name)) = name.split_once("::") {
                            type_name == enum_ty && variant == var_name
                        } else {
                            variant == name
                        };
                        if !name_matches { return false; }
                        // Arity check: bare `Variant` (no sub-pats) must match a no-field variant.
                        // `Variant(a, b)` must match the exact field count.
                        // Without this, `Color.Red` would match `Red(1, 2)` — wrong.
                        if sub_pats.is_empty() && !fields.is_empty() { return false; }
                        if !sub_pats.is_empty() && sub_pats.len() != fields.len() { return false; }
                        for (pat, field_val) in sub_pats.iter().zip(fields.iter()) {
                            if !self.match_pattern(pat, field_val, bindings) {
                                return false;
                            }
                        }
                        true
                    }
                    Value::Object(inner_rc) => {
                        // Struct destructuring: `Point(x, y)` matches a Point object,
                        // binding sub-patterns positionally to fields in declaration order.
                        let inner = inner_rc.borrow();
                        if inner.type_name != *name { return false; }
                        if sub_pats.len() != inner.fields.len() && !sub_pats.is_empty() { return false; }
                        for (pat, (_, field_val)) in sub_pats.iter().zip(inner.fields.iter()) {
                            if !self.match_pattern(pat, field_val, bindings) {
                                return false;
                            }
                        }
                        true
                    }
                    _ => false,
                }
            }
            Pattern::Tuple(sub_pats) => {
                let elems: &[Value] = match value {
                    Value::Tuple(v) => v,
                    Value::Array(v) => v,
                    _ => return false,
                };
                if sub_pats.len() != elems.len() { return false; }
                for (pat, elem) in sub_pats.iter().zip(elems.iter()) {
                    if !self.match_pattern(pat, elem, bindings) {
                        return false;
                    }
                }
                true
            }
        }
    }

    // ─── Ownership qualifier check ──────────────────────────────────────────

    /// Recursively verify that every user-defined named type (struct / enum) in a
    /// type annotation carries an explicit ownership qualifier.
    ///
    /// Validates that a type annotation is well-formed under boring's ownership model.
    ///
    /// With the new semantics (`T` = stack-owned, `T&` = borrow):
    ///
    /// Acceptable:
    ///   `User`         → Named("User")                     ✓  stack-owned (default)
    ///   `User&`        → Qualified(Named("User"), Borrow)  ✓  borrow (&User in Rust)
    ///   `User'`        → Qualified(Named("User"), Owned)   ✓  heap-owned (Box<User>)
    ///   `User'auto`    → Qualified(Named("User"), Auto)    ✓  Rc<User>
    ///   `User'shared`  → Qualified(Named("User"), Task)    ✓  Arc<User>
    ///   `User'?`       → Optional(Qualified(...))          ✓
    ///   `string`       → resolves to Qualified(Str, Task)  ✓  (primitive alias)
    ///   `int`          → resolves to Qualified(Int, Copy)  ✓
    ///   `T`, `K`       → TypeParam, deferred               ✓
    ///
    /// Not acceptable:
    ///   `Int` (bare)   → use `int` alias                   ✗
    ///   `Float` (bare) → use `float` alias                 ✗
    pub(crate) fn check_type_has_qualifier(&self, ty: &Type, line: usize) -> Result<(), Signal> {
        // Resolve aliases first (int → Qualified(Int,Copy), String → Str, …)
        let resolved = self.resolve_type(ty);
        self.check_resolved_qualifier(&resolved, line)
    }

    pub(crate) fn check_resolved_qualifier(&self, ty: &Type, line: usize) -> Result<(), Signal> {
        match ty {
            // Special types — always OK
            Type::Nil | Type::Void | Type::Never => Ok(()),

            // Bare primitive types: stack-allocated by default, always valid.
            Type::Int | Type::Uint | Type::Float | Type::Bool => Ok(()),
            // Bare String without qualifier: requires explicit qualification.
            Type::Str => Err(err("use 'string' instead of bare 'String' (which has no ownership qualifier)", line)),

            // Type parameters — resolved later, skip
            Type::TypeParam(_) => Ok(()),

            // Explicitly qualified — ownership is stated; no need to recurse into inner
            Type::Qualified(_, _) => Ok(()),

            // Bare named type → stack-owned (the default in boring).
            // `Dog d` = `Dog` on the stack, same as Rust's default.
            // Use `Dog&` for a borrow, `Dog'` / `Dog'heap` for Box<Dog>.
            Type::Named(_) => Ok(()),

            // Optional — check the wrapped type
            Type::Optional(inner) => self.check_resolved_qualifier(inner, line),

            // Collection element / key / value types
            Type::Array(elem) | Type::ArrayN(elem, _) | Type::Set(elem) => self.check_resolved_qualifier(elem, line),
            Type::Dict(k, v) => {
                self.check_resolved_qualifier(k, line)?;
                self.check_resolved_qualifier(v, line)
            }
            Type::Tuple(elems) => {
                for e in elems { self.check_resolved_qualifier(e, line)?; }
                Ok(())
            }

            // Function types — check param and return types
            Type::Fn(ret, params, _, _, _) => {
                if let Some(r) = ret { self.check_resolved_qualifier(r, line)?; }
                for p in params { self.check_resolved_qualifier(p, line)?; }
                Ok(())
            }

            // Generic applications — check type arguments
            Type::Generic(_, args) => {
                for a in args { self.check_resolved_qualifier(a, line)?; }
                Ok(())
            }

            // impl Trait — transparent, delegate to inner type
            Type::Dyn(inner) | Type::Impl(inner) => self.check_resolved_qualifier(inner, line),

            // Associated type reference — no ownership qualifier needed (resolved later)
            Type::SelfAssoc(_) | Type::AssocOf(_, _) => Ok(()),
        }
    }

    // ─── Type helpers ───────────────────────────────────────────────────────

    pub(crate) fn expect_bool(&self, val: Value, line: usize) -> Result<bool, Signal> {
        match val {
            Value::Bool(b) => Ok(b),
            other => Err(err(format!("expected Bool, got {}", other.type_name()), line)),
        }
    }

    pub(crate) fn expect_int(&self, val: Value, line: usize) -> Result<i64, Signal> {
        match val {
            Value::Int(n) => Ok(n),
            other => Err(err(format!("expected Int, got {}", other.type_name()), line)),
        }
    }

    pub(crate) fn expect_str(&self, val: Value, line: usize) -> Result<String, Signal> {
        match val {
            Value::Str(s) => Ok(s),
            other => Err(err(format!("expected Str, got {}", other.type_name()), line)),
        }
    }

    // ─── Generics helpers ──────────────────────────────────────────────────────

    /// Infer the boring type of a runtime value.
    pub(crate) fn type_of_value(val: &Value) -> Type {
        match val {
            Value::Int(_)   => Type::Int,
            Value::Uint(_)  => Type::Uint,
            Value::Float(_) => Type::Float,
            Value::Str(_)   => Type::Str,
            Value::Bool(_)  => Type::Bool,
            Value::Nil      => Type::Nil,
            Value::Void     => Type::Void,
            Value::Array(elems) => {
                let elem_ty = elems.first().map(|e| Self::type_of_value(e)).unwrap_or(Type::Nil);
                Type::Array(Box::new(elem_ty))
            }
            Value::Tuple(elems) => Type::Tuple(elems.iter().map(|e| Self::type_of_value(e)).collect()),
            Value::Dict(pairs) => {
                let k = pairs.first().map(|(k, _)| Self::type_of_value(k)).unwrap_or(Type::Nil);
                let v = pairs.first().map(|(_, v)| Self::type_of_value(v)).unwrap_or(Type::Nil);
                Type::Dict(Box::new(k), Box::new(v))
            }
            Value::Set(elems) => {
                let elem_ty = elems.first().map(|e| Self::type_of_value(e)).unwrap_or(Type::Nil);
                Type::Set(Box::new(elem_ty))
            }
            Value::Object(inner) => Type::Named(inner.borrow().type_name.clone()),
            Value::EnumVariant { type_name, .. } => Type::Named(type_name.clone()),
            Value::Future(_)  => Type::Generic("Future".into(), vec![]),
            Value::Range { .. } => Type::Named("Range".into()),
            _                 => Type::Named("_".into()),
        }
    }

    /// Recursively unify `param_ty` (a type annotation that may contain TypeParam / Named-as-param)
    /// with `actual_ty` (the runtime type of the actual argument) to extract bindings.
    /// Only names that appear in `type_params` (the function/struct's declared params) are bound.
    pub(crate) fn infer_from_type_params(
        type_params: &[String],
        param_ty: &Type,
        actual_ty: &Type,
        bindings: &mut HashMap<String, Type>,
    ) {
        match param_ty {
            // TypeParam("T") or Named("T") where T is in the declared type params list.
            // Guard: don't store a self-referential binding (T → Named("T")) — that would
            // cause resolve_type to loop forever when it finds the binding and re-resolves it.
            Type::TypeParam(name) if type_params.contains(name) => {
                let is_self_ref = matches!(actual_ty, Type::TypeParam(n) | Type::Named(n) if n == name);
                if !is_self_ref {
                    bindings.entry(name.clone()).or_insert_with(|| actual_ty.clone());
                }
            }
            Type::Named(name) if type_params.contains(name) => {
                let is_self_ref = matches!(actual_ty, Type::TypeParam(n) | Type::Named(n) if n == name);
                if !is_self_ref {
                    bindings.entry(name.clone()).or_insert_with(|| actual_ty.clone());
                }
            }
            Type::Optional(inner) => {
                // T? with a nil argument tells us nothing about T — skip to avoid
                // incorrectly binding T = Nil when another param will provide the real type.
                if matches!(actual_ty, Type::Nil) { return; }
                let inner_actual = match actual_ty {
                    Type::Optional(i) => i.as_ref().clone(),
                    other              => other.clone(),
                };
                Self::infer_from_type_params(type_params, inner, &inner_actual, bindings);
            }
            Type::Array(elem)   => {
                if let Type::Array(ae) = actual_ty {
                    Self::infer_from_type_params(type_params, elem, ae, bindings);
                }
            }
            Type::Set(elem)     => {
                if let Type::Set(ae) = actual_ty {
                    Self::infer_from_type_params(type_params, elem, ae, bindings);
                }
            }
            Type::Dict(k, v)    => {
                if let Type::Dict(ak, av) = actual_ty {
                    Self::infer_from_type_params(type_params, k, ak, bindings);
                    Self::infer_from_type_params(type_params, v, av, bindings);
                }
            }
            Type::Tuple(elems)  => {
                if let Type::Tuple(aelems) = actual_ty {
                    for (e, a) in elems.iter().zip(aelems.iter()) {
                        Self::infer_from_type_params(type_params, e, a, bindings);
                    }
                }
            }
            Type::Qualified(inner, _) => {
                // Strip qualifier and recurse (e.g. T'copy → T)
                Self::infer_from_type_params(type_params, inner, actual_ty, bindings);
            }
            Type::Generic(name, param_args) => {
                // Generic<A, B, ...> matched against Generic<X, Y, ...>
                if let Type::Generic(aname, actual_args) = actual_ty {
                    if name == aname && param_args.len() == actual_args.len() {
                        for (p, a) in param_args.iter().zip(actual_args.iter()) {
                            Self::infer_from_type_params(type_params, p, a, bindings);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Infer type-parameter bindings for a generic struct from its instantiated field values.
    pub(crate) fn infer_struct_type_params(decl: &StructDecl, obj_fields: &[(String, Value)]) -> HashMap<String, Type> {
        let mut bindings = HashMap::new();
        if decl.type_params.is_empty() { return bindings; }
        for field_decl in &decl.fields {
            if let Some((_, val)) = obj_fields.iter().find(|(n, _)| n == &field_decl.name) {
                Self::infer_from_type_params(
                    &decl.type_params,
                    &field_decl.ty,
                    &Self::type_of_value(val),
                    &mut bindings,
                );
            }
        }
        bindings
    }

    /// Verify that each `where T as Trait` constraint is satisfied by the inferred concrete type.
    /// Primitives (Int, Float, Str, Bool) skip enforcement — they are assumed to satisfy any trait.
    pub(crate) fn check_where_clause(
        &self,
        where_clause: &[(String, String)],
        bindings: &HashMap<String, Type>,
        line: usize,
    ) -> Result<(), Signal> {
        for (param, trait_name) in where_clause {
            let Some(concrete_ty) = bindings.get(param) else { continue };
            let base = strip_qualifiers(concrete_ty);
            // Primitives are assumed to satisfy any constraint
            match base {
                Type::Int | Type::Uint | Type::Float | Type::Str | Type::Bool => continue,
                _ => {}
            }
            let type_name = match base {
                Type::Named(n) => n.as_str(),
                _ => continue, // complex / anonymous types — skip
            };
            // Check if the struct satisfies the trait:
            // 1. Explicit protocols declaration in struct header
            // 2. Explicit conformance block
            // 3. Qualified method (e.g. `def Trait.method()`)
            // 4. Structural: all required trait methods are present in the struct
            let struct_val = self.global.borrow().get(type_name);
            let has_trait = match struct_val {
                Some(Value::Struct { decl, .. }) => {
                    if decl.protocols.iter().any(|p| p == trait_name) { true }
                    else if decl.methods.iter().any(|m| m.qualifier.as_deref() == Some(trait_name.as_str())) { true }
                    else {
                        // Structural: check all trait method signatures exist in struct methods
                        if let Some(trait_decl) = self.traits.get(trait_name.as_str()) {
                            let struct_method_names: std::collections::HashSet<&str> =
                                decl.methods.iter().map(|m| m.name.as_str()).collect();
                            trait_decl.signatures.iter().all(|sig| struct_method_names.contains(sig.name.as_str()))
                        } else {
                            false
                        }
                    }
                }
                Some(Value::EnumNamespace { methods, protocols, .. }) => {
                    if protocols.iter().any(|p| p == trait_name) { true }
                    else if methods.iter().any(|m| m.qualifier.as_deref() == Some(trait_name.as_str())) { true }
                    else {
                        if let Some(trait_decl) = self.traits.get(trait_name.as_str()) {
                            let method_names: std::collections::HashSet<&str> =
                                methods.iter().map(|m| m.name.as_str()).collect();
                            trait_decl.signatures.iter().all(|sig| method_names.contains(sig.name.as_str()))
                        } else {
                            false
                        }
                    }
                }
                _ => false,
            };
            if !has_trait {
                return Err(err(
                    format!("type '{}' does not conform to trait '{}'", type_name, trait_name),
                    line,
                ));
            }
        }
        Ok(())
    }

    // ─── Built-in `fs` module ─────────────────────────────────────────────────
    //
    // `fs.read("path")`, `fs.write(...)`, etc.  Uses synchronous std::fs in the
    // interpreter (no async runtime).  Errors are surfaced as Signal::Exception.

    pub(crate) fn call_fs_method(&mut self, method: &str, args: Vec<Value>, line: usize) -> Eval {
        // Helper: extract string argument at position `idx`.
        macro_rules! str_arg {
            ($idx:expr) => {
                match args.get($idx) {
                    Some(Value::Str(s)) => s.clone(),
                    Some(v) => format!("{}", v),
                    None => return Err(err(
                        format!("fs.{}: missing argument {}", method, $idx), line)),
                }
            };
        }
        macro_rules! fs_err {
            ($e:expr) => {
                return Err(Signal::Exception(Value::Str(format!("{}", $e))))
            };
        }

        match method {
            "read" => {
                let path = str_arg!(0);
                match std::fs::read_to_string(&path) {
                    Ok(s)  => Ok(Value::Str(s)),
                    Err(e) => fs_err!(e),
                }
            }
            "readLines" => {
                let path = str_arg!(0);
                match std::fs::read_to_string(&path) {
                    Ok(s) => {
                        let lines = s.lines()
                            .map(|l| Value::Str(l.to_string()))
                            .collect::<Vec<_>>();
                        Ok(Value::Array(lines))
                    }
                    Err(e) => fs_err!(e),
                }
            }
            "write" => {
                let path    = str_arg!(0);
                let content = str_arg!(1);
                match std::fs::write(&path, content.as_bytes()) {
                    Ok(())  => Ok(Value::Void),
                    Err(e)  => fs_err!(e),
                }
            }
            "append" => {
                use std::io::Write as _;
                let path    = str_arg!(0);
                let content = str_arg!(1);
                let result = std::fs::OpenOptions::new()
                    .append(true).create(true).open(&path)
                    .and_then(|mut f| f.write_all(content.as_bytes()));
                match result {
                    Ok(())  => Ok(Value::Void),
                    Err(e)  => fs_err!(e),
                }
            }
            "exists" => {
                let path = str_arg!(0);
                Ok(Value::Bool(std::path::Path::new(&path).exists()))
            }
            "isDir" => {
                let path = str_arg!(0);
                Ok(Value::Bool(std::path::Path::new(&path).is_dir()))
            }
            "isFile" => {
                let path = str_arg!(0);
                Ok(Value::Bool(std::path::Path::new(&path).is_file()))
            }
            "mkdir" => {
                let path = str_arg!(0);
                match std::fs::create_dir_all(&path) {
                    Ok(())  => Ok(Value::Void),
                    Err(e)  => fs_err!(e),
                }
            }
            "remove" => {
                let path = str_arg!(0);
                // Try file first; fall back to directory tree.
                let result = std::fs::remove_file(&path)
                    .or_else(|_| std::fs::remove_dir_all(&path));
                match result {
                    Ok(())  => Ok(Value::Void),
                    Err(e)  => fs_err!(e),
                }
            }
            "rename" | "move" => {
                let from = str_arg!(0);
                let to   = str_arg!(1);
                match std::fs::rename(&from, &to) {
                    Ok(())  => Ok(Value::Void),
                    Err(e)  => fs_err!(e),
                }
            }
            "copy" => {
                let from = str_arg!(0);
                let to   = str_arg!(1);
                match std::fs::copy(&from, &to) {
                    Ok(_)   => Ok(Value::Void),
                    Err(e)  => fs_err!(e),
                }
            }
            "list" => {
                let path = str_arg!(0);
                match std::fs::read_dir(&path) {
                    Ok(rd) => {
                        let mut entries = Vec::new();
                        for entry in rd {
                            match entry {
                                Ok(e) => entries.push(Value::Str(
                                    e.file_name().to_string_lossy().to_string())),
                                Err(e) => fs_err!(e),
                            }
                        }
                        Ok(Value::Array(entries))
                    }
                    Err(e) => fs_err!(e),
                }
            }
            "readBytes" => {
                let path = str_arg!(0);
                match std::fs::read(&path) {
                    Ok(bytes) => {
                        let arr = bytes.iter()
                            .map(|&b| Value::Int(b as i64))
                            .collect::<Vec<_>>();
                        Ok(Value::Array(arr))
                    }
                    Err(e) => fs_err!(e),
                }
            }
            "writeBytes" => {
                let path = str_arg!(0);
                let bytes = match args.get(1) {
                    Some(Value::Array(arr)) => arr.iter().map(|v| match v {
                        Value::Int(n) => *n as u8,
                        _ => 0u8,
                    }).collect::<Vec<u8>>(),
                    _ => return Err(err("fs.writeBytes: expected [int] as second argument", line)),
                };
                match std::fs::write(&path, &bytes) {
                    Ok(())  => Ok(Value::Void),
                    Err(e)  => fs_err!(e),
                }
            }
            other => Err(err(format!("fs.{}: unknown function", other), line)),
        }
    }
}
