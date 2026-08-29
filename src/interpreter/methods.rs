use super::*;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

impl Interpreter {
    pub(crate) fn call_method(&mut self, obj: Value, method: &str, args: Vec<Value>, line: usize, out_self: &mut Option<Value>) -> Eval {
        // Screen built-in method dispatch.
        if let Value::Screen { width: _width, height: _height, frame, resized, keys, pixels, title: _title, .. } = &obj {
            match method {
                // screen.present(pixels_array) — write pixel buffer; advance frame counter.
                // In simulation mode: store the pixels for PPM output.
                "present" => {
                    let buf = match args.into_iter().next() {
                        Some(Value::Array(v)) => v.iter().map(|x| match x {
                            Value::Uint(n) => *n as u32,
                            Value::Int(n)  => *n as u32,
                            _ => 0u32,
                        }).collect::<Vec<u32>>(),
                        _ => vec![],
                    };
                    *pixels.borrow_mut() = buf;
                    *frame.borrow_mut() += 1;
                    *resized.borrow_mut() = false;
                    return Ok(Value::Void);
                }
                // screen.key("q") — check whether key was pressed this frame.
                "key" => {
                    let key = match args.into_iter().next() {
                        Some(Value::Str(s)) => s.to_string(),
                        _ => return Ok(Value::Bool(false)),
                    };
                    let pressed = keys.borrow().contains(&key);
                    return Ok(Value::Bool(pressed));
                }
                _ => return Err(err(format!("Screen has no method '{}'", method), line)),
            }
        }
        // Screen property access as method call (screen.dimension(), screen.resized(), etc.).
        // These are also handled in field-access — keep both paths consistent.

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
                "cancel" | "abort" => {
                    // Cancellation is not supported in the interpreter — no-op (the
                    // task already ran to completion eagerly by the time this is
                    // called, since `task:` bodies aren't truly concurrent here).
                    // `abort` is JoinHandle's real name for this same operation
                    // (`future.abort()` in examples/todo.br's cancellation demo) —
                    // was missing here entirely, unlike its `cancel` synonym.
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
                    if let Value::Array(arr_rc) = obj {
                        let mut arr_owned = Value::rc_vec_into_owned(arr_rc);
                        if arr_owned.is_empty() {
                            // array.br declares `req T? pop(): native — nil if empty`,
                            // matching first()/last()'s Value::Nil-on-empty handling
                            // below — no element to remove, so out_self is left
                            // untouched (the array stays empty either way).
                            return Ok(Value::Nil);
                        }
                        let last = arr_owned.pop().unwrap();
                        *out_self = Some(Value::Array(arr_owned.into()));
                        return Ok(last);
                    }
                    unreachable!()
                }
                if method == "remove" {
                    if let Value::Array(arr_rc) = obj {
                        let mut arr_owned = Value::rc_vec_into_owned(arr_rc);
                        let idx_val = args.first().cloned().unwrap_or(Value::Int(0));
                        let idx = self.expect_int(idx_val, line)?;
                        let idx = if idx < 0 { arr_owned.len() as i64 + idx } else { idx };
                        let removed = if idx >= 0 && (idx as usize) < arr_owned.len() {
                            arr_owned.remove(idx as usize)
                        } else {
                            Value::Nil
                        };
                        *out_self = Some(Value::Array(arr_owned.into()));
                        return Ok(removed);
                    }
                    unreachable!()
                }
                // Methods recognized by call_array_method — safe to MOVE `obj` into it
                // rather than clone: since call_array_method always returns Some(_) for
                // these, we never fall through to the generic "no method" path below that
                // would otherwise need the original `obj`. Moving (instead of an Rc clone
                // that keeps this function's `obj` alive concurrently) lets the array's
                // Rc<Vec<Value>> reach a unique-owner refcount, so the mutation methods
                // (push, etc.) can rewrite the Vec in place instead of deep-cloning it.
                const KNOWN_ARRAY_METHODS: &[&str] = &[
                    "len", "length", "push", "contains", "first", "last", "reverse", "sort", "sortBy",
                    "map", "filter", "reduce", "join", "any", "all", "find", "indexOf", "flatMap", "flat",
                    "zip", "enumerate", "slice", "insert", "remove", "append", "count", "min", "max", "sum",
                    "isEmpty", "reversed", "sorted", "sortedBy", "take", "drop", "joined",
                    "firstIndex", "nextIndex", "removeAt", "getAt",
                ];
                if KNOWN_ARRAY_METHODS.contains(&method) {
                    let result = match self.call_array_method(obj, method, args, line)? {
                        Some(result) => result,
                        None => return Err(err(format!("no method '{}' on Array", method), line)),
                    };
                    const MUTATING: &[&str] = &["push", "append", "insert", "sort", "sortBy", "reverse", "removeAt"];
                    if MUTATING.contains(&method) {
                        if let Value::Array(_) = &result {
                            *out_self = Some(result.clone());
                        }
                    }
                    return Ok(result);
                }
                if let Some(result) = self.call_array_method(obj.clone(), method, args.clone(), line)? {
                    // For mutating array methods, set out_self so the caller can write back.
                    const MUTATING: &[&str] = &["push", "append", "insert", "sort", "sortBy", "reverse", "removeAt"];
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
            Value::Float64(f) => {
                let result = match method {
                    "sqrt"  => Some(Value::Float64(f.sqrt())),
                    "cbrt"  => Some(Value::Float64(f.cbrt())),
                    "abs"   => Some(Value::Float64(f.abs())),
                    "floor" => Some(Value::Float64(f.floor())),
                    "ceil"  => Some(Value::Float64(f.ceil())),
                    "round" => Some(Value::Float64(f.round())),
                    "exp"   => Some(Value::Float64(f.exp())),
                    "exp2"  => Some(Value::Float64(f.exp2())),
                    "ln"    => Some(Value::Float64(f.ln())),
                    "log2"  => Some(Value::Float64(f.log2())),
                    "log10" => Some(Value::Float64(f.log10())),
                    "sin"   => Some(Value::Float64(f.sin())),
                    "cos"   => Some(Value::Float64(f.cos())),
                    "tan"   => Some(Value::Float64(f.tan())),
                    "asin"  => Some(Value::Float64(f.asin())),
                    "acos"  => Some(Value::Float64(f.acos())),
                    "atan"  => Some(Value::Float64(f.atan())),
                    "sinh"  => Some(Value::Float64(f.sinh())),
                    "cosh"  => Some(Value::Float64(f.cosh())),
                    "tanh"  => Some(Value::Float64(f.tanh())),
                    "sign" | "signum" => Some(Value::Float64(f.signum())),
                    "isNaN" | "is_nan" => Some(Value::Bool(f.is_nan())),
                    "isInfinite" | "is_infinite" => Some(Value::Bool(f.is_infinite())),
                    "isFinite" | "is_finite" => Some(Value::Bool(f.is_finite())),
                    "toInt" | "int" => Some(Value::Int(*f as i64)),
                    "pow" | "powf" => {
                        let exp = args.first().cloned().unwrap_or(Value::Float64(1.0));
                        let e = match exp {
                            Value::Float64(e) => e,
                            Value::Int(n)   => n as f64,
                            _ => return Err(err("pow: argument must be a number", line)),
                        };
                        Some(Value::Float64(f.powf(e)))
                    }
                    "log" => {
                        let base = args.first().cloned().unwrap_or(Value::Float64(std::f64::consts::E));
                        let b = match base {
                            Value::Float64(b) => b,
                            Value::Int(n)   => n as f64,
                            _ => return Err(err("log: base must be a number", line)),
                        };
                        Some(Value::Float64(f.log(b)))
                    }
                    "atan2" => {
                        let other = args.first().cloned().unwrap_or(Value::Float64(0.0));
                        let o = match other {
                            Value::Float64(o) => o,
                            Value::Int(n)   => n as f64,
                            _ => return Err(err("atan2: argument must be a number", line)),
                        };
                        Some(Value::Float64(f.atan2(o)))
                    }
                    "clamp" => {
                        if args.len() < 2 { return Err(err("clamp: requires two arguments (min, max)", line)); }
                        let lo = match &args[0] { Value::Float64(v) => *v, Value::Int(n) => *n as f64, _ => return Err(err("clamp: min must be a number", line)) };
                        let hi = match &args[1] { Value::Float64(v) => *v, Value::Int(n) => *n as f64, _ => return Err(err("clamp: max must be a number", line)) };
                        Some(Value::Float64(f.clamp(lo, hi)))
                    }
                    _ => None,
                };
                if let Some(v) = result { return Ok(v); }
            }
            // Mirror of the Float64 block above, at f32 precision throughout —
            // Rust's f32 has the identical method surface to f64, so this is a
            // direct port, not new algorithm design (docs/float-width-types.md §8).
            Value::Float32(f) => {
                let result = match method {
                    "sqrt"  => Some(Value::Float32(f.sqrt())),
                    "cbrt"  => Some(Value::Float32(f.cbrt())),
                    "abs"   => Some(Value::Float32(f.abs())),
                    "floor" => Some(Value::Float32(f.floor())),
                    "ceil"  => Some(Value::Float32(f.ceil())),
                    "round" => Some(Value::Float32(f.round())),
                    "exp"   => Some(Value::Float32(f.exp())),
                    "exp2"  => Some(Value::Float32(f.exp2())),
                    "ln"    => Some(Value::Float32(f.ln())),
                    "log2"  => Some(Value::Float32(f.log2())),
                    "log10" => Some(Value::Float32(f.log10())),
                    "sin"   => Some(Value::Float32(f.sin())),
                    "cos"   => Some(Value::Float32(f.cos())),
                    "tan"   => Some(Value::Float32(f.tan())),
                    "asin"  => Some(Value::Float32(f.asin())),
                    "acos"  => Some(Value::Float32(f.acos())),
                    "atan"  => Some(Value::Float32(f.atan())),
                    "sinh"  => Some(Value::Float32(f.sinh())),
                    "cosh"  => Some(Value::Float32(f.cosh())),
                    "tanh"  => Some(Value::Float32(f.tanh())),
                    "sign" | "signum" => Some(Value::Float32(f.signum())),
                    "isNaN" | "is_nan" => Some(Value::Bool(f.is_nan())),
                    "isInfinite" | "is_infinite" => Some(Value::Bool(f.is_infinite())),
                    "isFinite" | "is_finite" => Some(Value::Bool(f.is_finite())),
                    "toInt" | "int" => Some(Value::Int(*f as i64)),
                    "pow" | "powf" => {
                        let exp = args.first().cloned().unwrap_or(Value::Float32(1.0));
                        let e = match exp {
                            Value::Float32(e) => e,
                            Value::Int(n)   => n as f32,
                            _ => return Err(err("pow: argument must be a number", line)),
                        };
                        Some(Value::Float32(f.powf(e)))
                    }
                    "log" => {
                        let base = args.first().cloned().unwrap_or(Value::Float32(std::f32::consts::E));
                        let b = match base {
                            Value::Float32(b) => b,
                            Value::Int(n)   => n as f32,
                            _ => return Err(err("log: base must be a number", line)),
                        };
                        Some(Value::Float32(f.log(b)))
                    }
                    "atan2" => {
                        let other = args.first().cloned().unwrap_or(Value::Float32(0.0));
                        let o = match other {
                            Value::Float32(o) => o,
                            Value::Int(n)   => n as f32,
                            _ => return Err(err("atan2: argument must be a number", line)),
                        };
                        Some(Value::Float32(f.atan2(o)))
                    }
                    "clamp" => {
                        if args.len() < 2 { return Err(err("clamp: requires two arguments (min, max)", line)); }
                        let lo = match &args[0] { Value::Float32(v) => *v, Value::Int(n) => *n as f32, _ => return Err(err("clamp: min must be a number", line)) };
                        let hi = match &args[1] { Value::Float32(v) => *v, Value::Int(n) => *n as f32, _ => return Err(err("clamp: max must be a number", line)) };
                        Some(Value::Float32(f.clamp(lo, hi)))
                    }
                    _ => None,
                };
                if let Some(v) = result { return Ok(v); }
            }
            Value::Channel { buf, is_sender, .. }
                if method == "send" => {
                    if !is_sender {
                        return Err(err("send called on a channel receiver", line));
                    }
                    let val = args.into_iter().next().unwrap_or(Value::Nil);
                    buf.borrow_mut().push_back(val);
                    return Ok(Value::Void);
                }
            // `receiver.recv()` — pop the oldest queued value, or `Nil` once drained.
            // Matches the documented `while let x = receiver.recv():` idiom (book.md's
            // "stop looping when the producer closes" shorthand): every sender in this
            // synchronous, single-threaded simulation has already run to completion
            // by the time a spawned `task:` consumer's recv loop starts (`task:`
            // bodies are evaluated eagerly, not truly concurrently — see
            // `Value::Future`'s doc), so the queue draining to empty is
            // indistinguishable from — and stands in for — a real closed channel.
            // This was previously entirely unimplemented (`send` had a dispatch arm,
            // `recv` never did), so every `receiver.recv()` call failed outright.
            Value::Channel { buf, is_sender, .. }
                if method == "recv" => {
                    if *is_sender {
                        return Err(err("recv called on a channel sender", line));
                    }
                    return Ok(buf.borrow_mut().pop_front().unwrap_or(Value::Nil));
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

        // `.clone()` — universal deep copy for all non-primitive types.
        if method == "clone" && args.is_empty() {
            return Ok(obj);
        }

        // std::sync::atomic::Atomic{Usize,Isize,U8,...,Bool} method dispatch.
        // The interpreter has no real threading, so these are modeled as a plain
        // single-field Object (see `construct_rust_type`, eval_expr.rs) holding the
        // current value; every fetch_*/load/store op below runs synchronously
        // against that one field. `Ordering` args (SeqCst, etc.) are accepted and
        // ignored — they're meaningless without real concurrent access.
        const ATOMIC_TYPES: &[&str] = &[
            "AtomicUsize", "AtomicIsize", "AtomicU8", "AtomicU16", "AtomicU32", "AtomicU64",
            "AtomicI8", "AtomicI16", "AtomicI32", "AtomicI64", "AtomicBool",
        ];
        if let Value::Object(inner_rc) = &obj {
            let type_name = inner_rc.borrow().type_name.clone();
            if ATOMIC_TYPES.contains(&type_name.as_str()) {
                let is_bool = type_name == "AtomicBool";
                match method {
                    "load" | "into_inner" | "get_mut" => {
                        let inner = inner_rc.borrow();
                        return Ok(inner.fields.iter().find(|(k, _)| k == "value")
                            .map(|(_, v)| v.clone()).unwrap_or(Value::Int(0)));
                    }
                    "store" => {
                        let new_val = args.into_iter().next().unwrap_or(Value::Int(0));
                        let mut inner = inner_rc.borrow_mut();
                        if let Some(f) = inner.fields.iter_mut().find(|(k, _)| k == "value") {
                            f.1 = new_val;
                        }
                        return Ok(Value::Void);
                    }
                    "fetch_add" | "fetch_sub" | "fetch_or" | "fetch_and" | "fetch_xor" | "swap" => {
                        let arg = args.into_iter().next().unwrap_or(Value::Int(0));
                        let mut inner = inner_rc.borrow_mut();
                        if let Some((_, cur)) = inner.fields.iter_mut().find(|(k, _)| k == "value") {
                            let old = cur.clone();
                            if is_bool {
                                let cur_b = matches!(cur, Value::Bool(true));
                                let arg_b = matches!(arg, Value::Bool(true));
                                *cur = Value::Bool(match method {
                                    "swap"      => arg_b,
                                    "fetch_or"  => cur_b || arg_b,
                                    "fetch_and" => cur_b && arg_b,
                                    "fetch_xor" => cur_b != arg_b,
                                    _ => cur_b,
                                });
                            } else {
                                let cur_i = self.expect_int(cur.clone(), line)?;
                                let arg_i = self.expect_int(arg, line)?;
                                *cur = Value::Int(match method {
                                    "fetch_add" => cur_i.wrapping_add(arg_i),
                                    "fetch_sub" => cur_i.wrapping_sub(arg_i),
                                    "fetch_or"  => cur_i | arg_i,
                                    "fetch_and" => cur_i & arg_i,
                                    "fetch_xor" => cur_i ^ arg_i,
                                    "swap"      => arg_i,
                                    _ => cur_i,
                                });
                            }
                            return Ok(old);
                        }
                        return Ok(Value::Int(0));
                    }
                    "compare_exchange" | "compare_exchange_weak" => {
                        let mut it = args.into_iter();
                        let expected = it.next().unwrap_or(Value::Int(0));
                        let new_val = it.next().unwrap_or(Value::Int(0));
                        let mut inner = inner_rc.borrow_mut();
                        if let Some((_, cur)) = inner.fields.iter_mut().find(|(k, _)| k == "value") {
                            let old = cur.clone();
                            return if Self::values_equal(cur, &expected) {
                                *cur = new_val;
                                Ok(Value::EnumVariant { type_name: "Result".into(), variant: "Ok".into(), fields: vec![old] })
                            } else {
                                Ok(Value::EnumVariant { type_name: "Result".into(), variant: "Err".into(), fields: vec![old] })
                            };
                        }
                        return Ok(Value::EnumVariant { type_name: "Result".into(), variant: "Err".into(), fields: vec![Value::Int(0)] });
                    }
                    _ => {}
                }
            }
        }

        // tokio::sync::Semaphore method dispatch — opaque in the interpreter (no real
        // concurrency to limit, so every permit is granted immediately). `acquire`/
        // `try_acquire` return `Nil`, matching the common `_ = semaphore.acquire()`
        // discard idiom; `available_permits` reports the semaphore as always-open.
        if let Value::Object(inner_rc) = &obj {
            if inner_rc.borrow().type_name == "Semaphore" {
                return match method {
                    "acquire" | "try_acquire" | "acquire_owned" | "try_acquire_owned" => Ok(Value::Nil),
                    "available_permits" => Ok(Value::Int(i64::MAX)),
                    "add_permits" | "close" | "forget_permits" => Ok(Value::Void),
                    "is_closed" => Ok(Value::Bool(false)),
                    _ => Err(err(format!("no method '{}' on Semaphore", method), line)),
                };
            }
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
            Type::Uint8  => matches!(val, Value::Uint8(_)),
            Type::Int8   => matches!(val, Value::Int8(_)),
            Type::Int16  => matches!(val, Value::Int16(_)),
            Type::Int32  => matches!(val, Value::Int32(_)),
            Type::Int64  => matches!(val, Value::Int64(_)),
            Type::Int128 => matches!(val, Value::Int128(_)),
            Type::Uint16 => matches!(val, Value::Uint16(_)),
            Type::Uint32 => matches!(val, Value::Uint32(_)),
            Type::Uint64 => matches!(val, Value::Uint64(_)),
            Type::Uint128 => matches!(val, Value::Uint128(_)),
            Type::Float32  => matches!(val, Value::Float32(_)),
            Type::Float64  => matches!(val, Value::Float64(_)),
            Type::Str    => matches!(val, Value::Str(_)),
            Type::Bool   => matches!(val, Value::Bool(_)),
            Type::Qualified(inner, _) => Self::value_matches_type_static(val, inner),
            Type::Named(name) => match name.as_str() {
                "int"    => matches!(val, Value::Int(_)),
                "uint"   => matches!(val, Value::Uint(_)),
                "uint8"  => matches!(val, Value::Uint8(_)),
                "int8"   => matches!(val, Value::Int8(_)),
                "int16"  => matches!(val, Value::Int16(_)),
                "int32"  => matches!(val, Value::Int32(_)),
                "int64"  => matches!(val, Value::Int64(_)),
                "int128" => matches!(val, Value::Int128(_)),
                "uint16" => matches!(val, Value::Uint16(_)),
                "uint32" => matches!(val, Value::Uint32(_)),
                "uint64" => matches!(val, Value::Uint64(_)),
                "uint128" => matches!(val, Value::Uint128(_)),
                "float32" | "f32" => matches!(val, Value::Float32(_)),
                "float" | "float64" | "f64" => matches!(val, Value::Float64(_)),
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
                let sub = self.expect_str(args.first().cloned().unwrap_or(Value::Nil), line)?;
                Ok(Some(Value::Bool(s.contains(sub.as_str()))))
            }
            "startsWith" => {
                let sub = self.expect_str(args.first().cloned().unwrap_or(Value::Nil), line)?;
                Ok(Some(Value::Bool(s.starts_with(sub.as_str()))))
            }
            "endsWith" => {
                let sub = self.expect_str(args.first().cloned().unwrap_or(Value::Nil), line)?;
                Ok(Some(Value::Bool(s.ends_with(sub.as_str()))))
            }
            "split" => {
                let sep = self.expect_str(args.first().cloned().unwrap_or(Value::Nil), line)?;
                let parts: Vec<Value> = s.split(sep.as_str()).map(|p| Value::Str(p.to_string())).collect();
                Ok(Some(Value::Array(parts.into())))
            }
            "trim" => Ok(Some(Value::Str(s.trim().to_string()))),
            "trimStart" => Ok(Some(Value::Str(s.trim_start().to_string()))),
            "trimEnd"   => Ok(Some(Value::Str(s.trim_end().to_string()))),
            "upper" | "toUpper" | "toUpperCase" | "uppercased" => Ok(Some(Value::Str(s.to_uppercase()))),
            "lower" | "toLower" | "toLowerCase" | "lowercased" => Ok(Some(Value::Str(s.to_lowercase()))),
            "replace" => {
                let from = self.expect_str(args.first().cloned().unwrap_or(Value::Nil), line)?;
                let to = self.expect_str(args.get(1).cloned().unwrap_or(Value::Nil), line)?;
                Ok(Some(Value::Str(s.replace(from.as_str(), to.as_str()))))
            }
            "chars" => {
                let chars: Vec<Value> = s.chars().map(|c| Value::Str(c.to_string())).collect();
                Ok(Some(Value::Array(chars.into())))
            }
            "lines" => {
                let lines: Vec<Value> = s.lines().map(|l| Value::Str(l.to_string())).collect();
                Ok(Some(Value::Array(lines.into())))
            }
            "slice" => {
                let char_count = s.chars().count();
                let start_i = self.expect_int(args.first().cloned().unwrap_or(Value::Int(0)), line)?;
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
                let n = self.expect_int(args.first().cloned().unwrap_or(Value::Int(0)), line)? as usize;
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
                    Ok(f) => Ok(Some(Value::Float64(f))),
                    Err(_) => Ok(Some(Value::Nil)),
                }
            }
            "parseFloat32" => {
                match s.trim().parse::<f32>() {
                    Ok(f) => Ok(Some(Value::Float32(f))),
                    Err(_) => Ok(Some(Value::Nil)),
                }
            }
            "indexOf" => {
                let sub = self.expect_str(args.first().cloned().unwrap_or(Value::Nil), line)?;
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
                let mut new_arr = Value::rc_vec_into_owned(arr);
                new_arr.push(args.into_iter().next().unwrap_or(Value::Nil));
                Ok(Some(Value::Array(new_arr.into())))
            }
            "contains" => {
                let target = args.into_iter().next().unwrap_or(Value::Nil);
                Ok(Some(Value::Bool(arr.contains(&target))))
            }
            "first" => Ok(Some(arr.first().cloned().unwrap_or(Value::Nil))),
            "last" => Ok(Some(arr.last().cloned().unwrap_or(Value::Nil))),
            "reverse" => {
                let mut new_arr = Value::rc_vec_into_owned(arr);
                new_arr.reverse();
                Ok(Some(Value::Array(new_arr.into())))
            }
            "sort" => {
                let mut new_arr = Value::rc_vec_into_owned(arr);
                new_arr.sort_by(|a, b| {
                    match (a, b) {
                        (Value::Int(x), Value::Int(y)) => x.cmp(y),
                        (Value::Float64(x), Value::Float64(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Float32(x), Value::Float32(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Str(x), Value::Str(y)) => x.cmp(y),
                        _ => std::cmp::Ordering::Equal,
                    }
                });
                Ok(Some(Value::Array(new_arr.into())))
            }
            "sortBy" => {
                // sortBy (elem): key_expr  — sort by a key extractor closure (ascending).
                // Negate the key for descending: sortBy (e): -e.score
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                let arr = Value::rc_vec_into_owned(arr);
                let mut keyed: Vec<(Value, Value)> = Vec::with_capacity(arr.len());
                for item in arr {
                    let key = self.call_value(closure.clone(), vec![item.clone()], line, false)?;
                    keyed.push((key, item));
                }
                keyed.sort_by(|(ka, _), (kb, _)| {
                    match (ka, kb) {
                        (Value::Int(a), Value::Int(b))     => a.cmp(b),
                        (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Float32(a), Value::Float32(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Int(a), Value::Float64(b))   => (*a as f64).partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Float64(a), Value::Int(b))   => a.partial_cmp(&(*b as f64)).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Int(a), Value::Float32(b))   => (*a as f32).partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Float32(a), Value::Int(b))   => a.partial_cmp(&(*b as f32)).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Str(a), Value::Str(b))     => a.cmp(b),
                        _                                   => std::cmp::Ordering::Equal,
                    }
                });
                let sorted: Vec<Value> = keyed.into_iter().map(|(_, v)| v).collect();
                Ok(Some(Value::Array(sorted.into())))
            }
            "map" => {
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                let mut result = Vec::new();
                for item in Value::rc_vec_into_owned(arr) {
                    let r = self.call_value(closure.clone(), vec![item], line, false)?;
                    result.push(r);
                }
                Ok(Some(Value::Array(result.into())))
            }
            "filter" => {
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                let mut result = Vec::new();
                for item in Value::rc_vec_into_owned(arr) {
                    let r = self.call_value(closure.clone(), vec![item.clone()], line, false)?;
                    let b = self.expect_bool(r, line)?;
                    if b { result.push(item); }
                }
                Ok(Some(Value::Array(result.into())))
            }
            "reduce" => {
                // reduce(init, closure) — initial value first, closure second
                let init = args.first().cloned().unwrap_or(Value::Nil);
                let closure = args.get(1).cloned().unwrap_or(Value::Nil);
                let mut acc = init;
                for item in Value::rc_vec_into_owned(arr) {
                    acc = self.call_value(closure.clone(), vec![acc, item], line, false)?;
                }
                Ok(Some(acc))
            }
            "join" => {
                let sep = match args.first() {
                    Some(Value::Str(s)) => s.clone(),
                    _ => String::new(),
                };
                let parts: Vec<String> = arr.iter().map(|v| format!("{}", v)).collect();
                Ok(Some(Value::Str(parts.join(&sep))))
            }
            "any" => {
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                for item in Value::rc_vec_into_owned(arr) {
                    let r = self.call_value(closure.clone(), vec![item], line, false)?;
                    if self.expect_bool(r, line)? { return Ok(Some(Value::Bool(true))); }
                }
                Ok(Some(Value::Bool(false)))
            }
            "all" => {
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                for item in Value::rc_vec_into_owned(arr) {
                    let r = self.call_value(closure.clone(), vec![item], line, false)?;
                    if !self.expect_bool(r, line)? { return Ok(Some(Value::Bool(false))); }
                }
                Ok(Some(Value::Bool(true)))
            }
            "find" => {
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                for item in Value::rc_vec_into_owned(arr) {
                    let r = self.call_value(closure.clone(), vec![item.clone()], line, false)?;
                    if self.expect_bool(r, line)? { return Ok(Some(item)); }
                }
                Ok(Some(Value::Nil))
            }
            "indexOf" => {
                let target = args.into_iter().next().unwrap_or(Value::Nil);
                // `indexOf((t): pred)` — a closure/predicate argument (same shorthand
                // `.find()` above accepts) finds by predicate, not by equality: a
                // `Value::Closure`/`Fn`/`NativeFn` never equals a `Task`/etc. element via
                // `==`, so the equality branch below would silently always return `Nil`
                // for this call shape (confirmed via examples/todo.br's `complete_task`,
                // which always threw "not found" before this fix).
                if matches!(target, Value::Closure { .. } | Value::Fn { .. } | Value::NativeFn { .. } | Value::OverloadedFn { .. }) {
                    for (i, item) in Value::rc_vec_into_owned(arr).into_iter().enumerate() {
                        let r = self.call_value(target.clone(), vec![item], line, false)?;
                        if self.expect_bool(r, line)? { return Ok(Some(Value::Int(i as i64))); }
                    }
                    return Ok(Some(Value::Nil));
                }
                for (i, item) in arr.iter().enumerate() {
                    if item == &target { return Ok(Some(Value::Int(i as i64))); }
                }
                Ok(Some(Value::Nil))
            }
            "flatMap" => {
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                let mut result = Vec::new();
                for item in Value::rc_vec_into_owned(arr) {
                    let r = self.call_value(closure.clone(), vec![item], line, false)?;
                    match r {
                        Value::Array(inner) => result.extend(Value::rc_vec_into_owned(inner)),
                        other => result.push(other),
                    }
                }
                Ok(Some(Value::Array(result.into())))
            }
            "flat" => {
                let mut result = Vec::new();
                for item in Value::rc_vec_into_owned(arr) {
                    match item {
                        Value::Array(inner) => result.extend(Value::rc_vec_into_owned(inner)),
                        other => result.push(other),
                    }
                }
                Ok(Some(Value::Array(result.into())))
            }
            "zip" => {
                let other = match args.into_iter().next() {
                    Some(Value::Array(a)) => Value::rc_vec_into_owned(a),
                    _ => vec![],
                };
                let result: Vec<Value> = Value::rc_vec_into_owned(arr).into_iter().zip(other)
                    .map(|(a, b)| Value::Tuple(vec![a, b]))
                    .collect();
                Ok(Some(Value::Array(result.into())))
            }
            "enumerate" => {
                let result: Vec<Value> = Value::rc_vec_into_owned(arr).into_iter().enumerate()
                    .map(|(i, v)| Value::Tuple(vec![Value::Int(i as i64), v]))
                    .collect();
                Ok(Some(Value::Array(result.into())))
            }
            "slice" => {
                let len = arr.len();
                let start_i = self.expect_int(args.first().cloned().unwrap_or(Value::Int(0)), line)?;
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
                Ok(Some(Value::Array(arr[start..end.max(start)].to_vec().into())))
            }
            "insert" => {
                let idx_i = self.expect_int(args.first().cloned().unwrap_or(Value::Int(0)), line)?;
                let val = args.get(1).cloned().unwrap_or(Value::Nil);
                let mut new_arr = Value::rc_vec_into_owned(arr);
                let idx = if idx_i < 0 {
                    (new_arr.len() as i64 + idx_i).max(0) as usize
                } else {
                    (idx_i as usize).min(new_arr.len())
                };
                new_arr.insert(idx, val);
                Ok(Some(Value::Array(new_arr.into())))
            }
            "remove" => {
                let idx = self.expect_int(args.first().cloned().unwrap_or(Value::Int(0)), line)?;
                let mut new_arr = Value::rc_vec_into_owned(arr);
                let idx = if idx < 0 { new_arr.len() as i64 + idx } else { idx };
                if idx >= 0 && (idx as usize) < new_arr.len() {
                    new_arr.remove(idx as usize);
                }
                Ok(Some(Value::Array(new_arr.into())))
            }
            "append" => {
                let other = match args.into_iter().next() {
                    Some(Value::Array(a)) => Value::rc_vec_into_owned(a),
                    _ => vec![],
                };
                let mut new_arr = Value::rc_vec_into_owned(arr);
                new_arr.extend(other);
                Ok(Some(Value::Array(new_arr.into())))
            }
            "count" => {
                match args.into_iter().next() {
                    Some(closure @ (Value::Closure { .. } | Value::Fn { .. } | Value::NativeFn { .. })) => {
                        let mut n = 0i64;
                        for item in Value::rc_vec_into_owned(arr) {
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
                    (Value::Float64(x), Value::Float64(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                    (Value::Float32(x), Value::Float32(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                    _ => std::cmp::Ordering::Equal,
                }).cloned().unwrap_or(Value::Nil);
                Ok(Some(result))
            }
            "max" => {
                let result = arr.iter().max_by(|a, b| match (a, b) {
                    (Value::Int(x), Value::Int(y)) => x.cmp(y),
                    (Value::Float64(x), Value::Float64(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                    (Value::Float32(x), Value::Float32(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                    _ => std::cmp::Ordering::Equal,
                }).cloned().unwrap_or(Value::Nil);
                Ok(Some(result))
            }
            "sum" => {
                let mut int_sum = 0i64;
                let mut float_sum = 0.0f64;
                let mut has_float = false;
                for item in arr.iter() {
                    match item {
                        Value::Int(n) => int_sum += n,
                        Value::Float64(f) => { float_sum += f; has_float = true; }
                        Value::Float32(f) => { float_sum += *f as f64; has_float = true; }
                        _ => {}
                    }
                }
                if has_float {
                    Ok(Some(Value::Float64(int_sum as f64 + float_sum)))
                } else {
                    Ok(Some(Value::Int(int_sum)))
                }
            }
            "isEmpty" => Ok(Some(Value::Bool(arr.is_empty()))),
            "reversed" => {
                let mut new_arr = Value::rc_vec_into_owned(arr);
                new_arr.reverse();
                Ok(Some(Value::Array(new_arr.into())))
            }
            "sorted" => {
                let mut new_arr = Value::rc_vec_into_owned(arr);
                new_arr.sort_by(|a, b| match (a, b) {
                    (Value::Int(x), Value::Int(y)) => x.cmp(y),
                    (Value::Float64(x), Value::Float64(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                    (Value::Float32(x), Value::Float32(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                    (Value::Str(x), Value::Str(y)) => x.cmp(y),
                    _ => std::cmp::Ordering::Equal,
                });
                Ok(Some(Value::Array(new_arr.into())))
            }
            "sortedBy" => {
                let closure = args.into_iter().next().unwrap_or(Value::Nil);
                let mut new_arr = Value::rc_vec_into_owned(arr);
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
                        (Value::Float64(x), Value::Float64(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Float32(x), Value::Float32(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Str(x), Value::Str(y)) => x.cmp(y),
                        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
                        _ => std::cmp::Ordering::Equal,
                    }
                });
                let sorted: Vec<Value> = indices.into_iter().map(|i| new_arr[i].clone()).collect();
                Ok(Some(Value::Array(sorted.into())))
            }
            "take" => {
                let n = self.expect_int(args.first().cloned().unwrap_or(Value::Int(0)), line)?;
                let n = (n.max(0)) as usize;
                Ok(Some(Value::Array(Value::rc_vec_into_owned(arr).into_iter().take(n).collect::<Vec<_>>().into())))
            }
            "drop" => {
                let n = self.expect_int(args.first().cloned().unwrap_or(Value::Int(0)), line)?;
                let n = (n.max(0)) as usize;
                Ok(Some(Value::Array(Value::rc_vec_into_owned(arr).into_iter().skip(n).collect::<Vec<_>>().into())))
            }
            "joined" => {
                let sep = match args.first() {
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
                        let mut new_arr = Value::rc_vec_into_owned(arr);
                        new_arr.remove(pos);
                        Ok(Some(Value::Array(new_arr.into())))
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
            "keys" => Ok(Some(Value::Array(pairs.into_iter().map(|(k, _)| k).collect::<Vec<_>>().into()))),
            "values" => Ok(Some(Value::Array(pairs.into_iter().map(|(_, v)| v).collect::<Vec<_>>().into()))),
            "len" => Ok(Some(Value::Int(pairs.len() as i64))),
            "contains" | "containsKey" | "has" => {
                let key = args.first().cloned().unwrap_or(Value::Nil);
                Ok(Some(Value::Bool(pairs.iter().any(|(k, _)| k == &key))))
            }
            "get" => {
                let key = args.first().cloned().unwrap_or(Value::Nil);
                let default = args.get(1).cloned().unwrap_or(Value::Nil);
                let found = pairs.into_iter().find(|(k, _)| k == &key).map(|(_, v)| v);
                Ok(Some(found.unwrap_or(default)))
            }
            "remove" => {
                let key = args.first().cloned().unwrap_or(Value::Nil);
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
                let key = args.first().cloned().unwrap_or(Value::Nil);
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
            "toArray" => Ok(Some(Value::Array(set.into()))),
            "isEmpty" => Ok(Some(Value::Bool(set.is_empty()))),
            "count" | "len" | "length" => Ok(Some(Value::Int(set.len() as i64))),
            "union" => {
                let other = match args.into_iter().next() {
                    Some(Value::Set(s)) => s,
                    Some(Value::Array(a)) => Value::rc_vec_into_owned(a),
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
                    Some(Value::Array(a)) => Value::rc_vec_into_owned(a),
                    _ => vec![],
                };
                let new_set: Vec<Value> = set.into_iter().filter(|v| other.contains(v)).collect();
                Ok(Some(Value::Set(new_set)))
            }
            "difference" => {
                let other = match args.into_iter().next() {
                    Some(Value::Set(s)) => s,
                    Some(Value::Array(a)) => Value::rc_vec_into_owned(a),
                    _ => vec![],
                };
                let new_set: Vec<Value> = set.into_iter().filter(|v| !other.contains(v)).collect();
                Ok(Some(Value::Set(new_set)))
            }
            "isSubset" => {
                let other = match args.into_iter().next() {
                    Some(Value::Set(s)) => s,
                    Some(Value::Array(a)) => Value::rc_vec_into_owned(a),
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
            // Screen property access.
            Value::Screen { ref width, ref height, ref frame, ref resized, ref created_at, .. } => {
                match field {
                    "dimension" => {
                        let w = *width.borrow();
                        let h = *height.borrow();
                        let inner = crate::interpreter::ObjectInner {
                            type_name: "Dimension".into(),
                            fields: vec![
                                ("width".into(),  Value::Uint(w)),
                                ("height".into(), Value::Uint(h)),
                            ],
                        };
                        Ok(Value::Object(Rc::new(RefCell::new(inner))))
                    }
                    "resized" => Ok(Value::Bool(*resized.borrow())),
                    "frame"   => Ok(Value::Uint(*frame.borrow())),
                    "width"   => Ok(Value::Uint(*width.borrow())),
                    "height"  => Ok(Value::Uint(*height.borrow())),
                    // "seconds elapsed since loop start" (docs/gpu-display.md) —
                    // was entirely unimplemented; see the `created_at` field's doc.
                    "time"    => Ok(Value::Float64(created_at.elapsed().as_secs_f64())),
                    other => Err(err(format!("Screen has no field '{}'", other), line)),
                }
            }
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

    pub(crate) fn get_index(&mut self, obj: Value, idx: Value, line: usize, col: usize, len: usize) -> Eval {
        match obj {
            Value::Array(arr) => {
                let pos: usize = match idx {
                    Value::Index(IndexValue::Array(p)) => p,
                    other => {
                        let i = self.expect_int(other, line)?;
                        let i = if i < 0 { arr.len() as i64 + i } else { i };
                        if i < 0 || i as usize >= arr.len() {
                            return Err(err_span(format!("array index {} out of bounds (len {})", i, arr.len()), line, col, len));
                        }
                        i as usize
                    }
                };
                arr.get(pos).cloned()
                    .ok_or_else(|| err_span(format!("array index {} out of bounds (len {})", pos, arr.len()), line, col, len))
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
                            .ok_or_else(|| err_span(format!("set index {} out of bounds (len {})", pos, set.len()), line, col, len))
                    }
                    _ => Err(err_span("set subscript requires a set index — use firstIndex() / nextIndex()", line, col, len)),
                }
            }
            Value::Str(s) => {
                let i = self.expect_int(idx, line)?;
                let chars: Vec<char> = s.chars().collect();
                let i = if i < 0 { chars.len() as i64 + i } else { i };
                if i < 0 || i as usize >= chars.len() {
                    Err(err_span(format!("string index {} out of bounds", i), line, col, len))
                } else {
                    Ok(Value::Str(chars[i as usize].to_string()))
                }
            }
            other => Err(err_span(format!("cannot index into {}", other.type_name()), line, col, len)),
        }
    }

    /// Fast path for `var_name[idx] = val` on a plain local variable holding an
    /// array. The generic path (in `assign`, below) reads the array via
    /// `eval_expr` (cloning the `Rc<Vec<Value>>` out of the env slot while the
    /// slot's own copy stays alive), so copy-on-write in `Value::rc_vec_into_owned`
    /// always has to deep-clone — the same O(n)-per-write issue `.push()` had (see
    /// `try_fast_mutating_array_call`'s doc comment). Taking the value out of the
    /// slot instead drops that to one owner in the common unaliased case, making
    /// index-assignment O(1) instead of O(n) per write (so a loop writing n
    /// elements is O(n), not O(n^2)) — the mel-spectrogram/matrix code this
    /// interpreter runs writes every element of large flat arrays this way.
    ///
    /// `idx_expr` is evaluated first (while the slot is still intact) so a
    /// self-referential index like `arr[arr.length - 1] = v` still sees the real
    /// array. Returns `None` if `name` doesn't currently hold an array — the
    /// caller falls through to the generic path unchanged, without having
    /// evaluated `idx_expr` (no risk of double evaluation).
    #[inline(never)]
    fn try_fast_array_index_assign(
        &mut self,
        name: &str,
        idx_expr: &Expr,
        val: Value,
        env: &EnvRef,
        line: usize,
    ) -> Option<Result<(), Signal>> {
        if !matches!(env.borrow().get(name), Some(Value::Array(_))) {
            return None;
        }
        // `'sync` fields need to write through to block-shared storage (see
        // `Interpreter::sync_fields`) — bail out to `assign`'s slower Index path,
        // which handles them explicitly, instead of writing to this thread's own
        // private `env` copy where no other thread in the block would see it.
        if self.sync_fields.contains_key(name) {
            return None;
        }
        Some((|| {
            let idx = self.eval_expr(idx_expr, Rc::clone(env))?;
            let taken = env.borrow_mut().take(name).unwrap_or(Value::Nil);
            let arr_rc = match taken {
                Value::Array(rc) => rc,
                other => {
                    // Extremely unlikely (idx evaluation somehow changed the
                    // variable's type mid-expression) — restore and report clearly.
                    env.borrow_mut().force_set(name, other);
                    return Err(err(format!("'{}' is no longer an array", name), line));
                }
            };
            let mut arr = Value::rc_vec_into_owned(arr_rc);
            let pos_result: Result<usize, Signal> = match idx {
                Value::Index(IndexValue::Array(p)) => Ok(p),
                other => {
                    let i = self.expect_int(other, line)?;
                    let i = if i < 0 { arr.len() as i64 + i } else { i };
                    if i < 0 || i as usize >= arr.len() {
                        Err(err(format!("array index {} out of bounds (len {})", i, arr.len()), line))
                    } else {
                        Ok(i as usize)
                    }
                }
            };
            match pos_result {
                Ok(pos) => {
                    if pos < arr.len() { arr[pos] = val; }
                    env.borrow_mut().force_set(name, Value::Array(arr.into()));
                    Ok(())
                }
                Err(e) => {
                    env.borrow_mut().force_set(name, Value::Array(arr.into()));
                    Err(e)
                }
            }
        })())
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
                            let self_expr = Expr { kind: ExprKind::Var("self".to_string()), line, col: 0, len: 0 };
                            let field_expr = Expr { kind: ExprKind::Field(Box::new(self_expr), name.clone()), line, col: 0, len: 0 };
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
                // Check: cannot mutate a let binding's field.
                // `self` is excluded — it's always bound via `define_mut` (see
                // every `fn_env.define_mut("self", ...)` call site) regardless of
                // `def`/`req`, so this never distinguished self-field-write
                // legality anyway; that's `self.current_method_mutating` below
                // ("cannot mutate non-transient field from a req method").
                if let ExprKind::Var(binding_name) = &obj_expr.kind {
                    if binding_name != "self" && !env.borrow().is_content_mutable(binding_name) {
                        return Err(err(
                            format!("cannot mutate field '{}' on non-mut binding '{}' — declare it with `mut` or `var mut` to permit content mutation", field, binding_name),
                            line,
                        ));
                    }
                    // `var T'shared` is reassignable but has no interior mutability
                    // (Arc<T> — same restriction as calling a mutating method on it).
                    if !env.borrow().is_actor(binding_name) && env.borrow().is_shared(binding_name) {
                        return Err(err(
                            format!("cannot assign to field '{}' on shared binding '{}' — use T'actor for interior mutability", field, binding_name),
                            line,
                        ));
                    }
                }
                // `arr[i].field = v` — same permission as `arr[i].method()`
                // (see `eval_expr.rs`'s identical check): the collection's own
                // declared element type must grant `mut` — `[mut Point] arr`
                // vs plain `[Point] arr` (docs/book.md).
                if let ExprKind::Index(inner_obj, _idx) = &obj_expr.kind {
                    if let ExprKind::Var(coll_name) = &inner_obj.kind {
                        if let Some(coll_ty) = env.borrow().get_declared_type(coll_name) {
                            if let Some(elem_ty) = coll_ty.index_element_type() {
                                if !elem_ty.grants_mut() {
                                    return Err(err(
                                        format!("cannot assign to field '{}' on an element of '{}' — its declared element type doesn't grant content mutation; declare it `[mut T]`/`{{K = mut V}}`", field, coll_name),
                                        line,
                                    ));
                                }
                            }
                        }
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
                        Ok(())
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
                        Err(err(format!("enum variant has no settable field '{}'", field), line))
                    }
                    _ => Err(err("cannot assign field on non-object", line)),
                }
            }
            ExprKind::Index(obj_expr, idx_expr) => {
                // Fast path — see `try_fast_array_index_assign`'s doc comment. Kept in
                // its own #[inline(never)] function so this stays out of `assign`'s
                // (recursive) stack frame — see `try_fast_mutating_array_call` for why
                // that matters in this interpreter's debug-build stack budget.
                if let ExprKind::Var(name) = &obj_expr.kind {
                    if let Some(result) = self.try_fast_array_index_assign(name, idx_expr, val.clone(), &env, line) {
                        return result;
                    }
                    // `'sync` fields: write straight into the block-shared backing
                    // array instead of this thread's own private env copy, so every
                    // other thread in the block observes it once they cross the next
                    // barrier. See `Interpreter::sync_fields`'s doc comment.
                    if let Some(shared) = self.sync_fields.get(name).cloned() {
                        let idx = self.eval_expr(idx_expr, Rc::clone(&env))?;
                        let i = self.expect_int(idx, line)?;
                        let mut arr = shared.lock().unwrap();
                        let pos = if i < 0 { arr.len() as i64 + i } else { i };
                        if pos < 0 || pos as usize >= arr.len() {
                            return Err(err(format!("array index {} out of bounds (len {})", i, arr.len()), line));
                        }
                        let Some(tv) = super::eval_gpu::to_thread_value(&val) else {
                            return Err(err("value is not valid 'sync field data", line));
                        };
                        arr[pos as usize] = tv;
                        return Ok(());
                    }
                }
                // Evaluate the object before the index — matches the real evaluation
                // order of `obj[idx] = val` in the transpiled Rust (and left-to-right
                // evaluation in general), so side effects in `obj`/`idx` fire in the
                // order the source implies.
                let obj = self.eval_expr(obj_expr, Rc::clone(&env))?;
                let idx = self.eval_expr(idx_expr, Rc::clone(&env))?;
                match obj {
                    Value::Array(arr_rc) => {
                        let mut arr = Value::rc_vec_into_owned(arr_rc);
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
                        self.assign(obj_expr, Value::Array(arr.into()), env, line)?;
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
            if let Ok(Value::Str(s)) = self.cast_value(val.clone(), &str_ty, line) { return Ok(s) }
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
                                    "e"  => match val {
                                        Value::Float64(f) => format!("{:e}", f),
                                        Value::Float32(f) => format!("{:e}", f),
                                        _ => format!("{}", val),
                                    },
                                    "E"  => match val {
                                        Value::Float64(f) => format!("{:E}", f),
                                        Value::Float32(f) => format!("{:E}", f),
                                        _ => format!("{}", val),
                                    },
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

    #[allow(clippy::only_used_in_recursion)]
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
                    type_matches(strip_qualifiers(&self.resolve_type(&a.ty)), strip_qualifiers(ty))
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
                    type_matches(strip_qualifiers(&self.resolve_type(&a.ty)), strip_qualifiers(ty))
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
            Type::Qualified(inner, _) => self.cast_value(val, inner, line),
            // `as int`/`as uint8`/`as int32`/... — every numeric kind is a valid source for
            // every other numeric kind's cast target (mirrors Rust's own `as` cast, which is
            // exactly the escape hatch users reach for since direct arithmetic between two
            // *distinct* fixed-width kinds is otherwise a type error). One macro arm covers
            // the full 13x13 matrix (12 numeric kinds + bool/string) instead of hand-listing
            // each pair, using `TryFrom` (implemented by std for every integer-width pair).
            Type::Int | Type::Uint | Type::Uint8
                | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128 => {
                macro_rules! cast_to {
                    ($Variant:ident, $ty:ty) => {
                        Ok(match val {
                            Value::Int(n) => <$ty>::try_from(n).map(Value::$Variant).unwrap_or(Value::Nil),
                            Value::Uint(n) => <$ty>::try_from(n).map(Value::$Variant).unwrap_or(Value::Nil),
                            Value::Uint8(n) => <$ty>::try_from(n).map(Value::$Variant).unwrap_or(Value::Nil),
                            Value::Int8(n) => <$ty>::try_from(n).map(Value::$Variant).unwrap_or(Value::Nil),
                            Value::Int16(n) => <$ty>::try_from(n).map(Value::$Variant).unwrap_or(Value::Nil),
                            Value::Int32(n) => <$ty>::try_from(n).map(Value::$Variant).unwrap_or(Value::Nil),
                            Value::Int64(n) => <$ty>::try_from(n).map(Value::$Variant).unwrap_or(Value::Nil),
                            Value::Int128(n) => <$ty>::try_from(n).map(Value::$Variant).unwrap_or(Value::Nil),
                            Value::Uint16(n) => <$ty>::try_from(n).map(Value::$Variant).unwrap_or(Value::Nil),
                            Value::Uint32(n) => <$ty>::try_from(n).map(Value::$Variant).unwrap_or(Value::Nil),
                            Value::Uint64(n) => <$ty>::try_from(n).map(Value::$Variant).unwrap_or(Value::Nil),
                            Value::Uint128(n) => <$ty>::try_from(n).map(Value::$Variant).unwrap_or(Value::Nil),
                            Value::Float64(f) => {
                                if f >= <$ty>::MIN as f64 && f <= <$ty>::MAX as f64 { Value::$Variant(f as $ty) } else { Value::Nil }
                            }
                            Value::Float32(f) => {
                                if f as f64 >= <$ty>::MIN as f64 && f as f64 <= <$ty>::MAX as f64 { Value::$Variant(f as $ty) } else { Value::Nil }
                            }
                            Value::Str(ref s) => s.trim().parse::<$ty>().map(Value::$Variant).unwrap_or(Value::Nil),
                            Value::Bool(b) => Value::$Variant(if b { 1 as $ty } else { 0 as $ty }),
                            _ => Value::Nil,
                        })
                    };
                }
                match ty {
                    Type::Int => cast_to!(Int, i64),
                    Type::Uint => cast_to!(Uint, u64),
                    Type::Uint8 => cast_to!(Uint8, u8),
                    Type::Int8 => cast_to!(Int8, i8),
                    Type::Int16 => cast_to!(Int16, i16),
                    Type::Int32 => cast_to!(Int32, i32),
                    Type::Int64 => cast_to!(Int64, i64),
                    Type::Int128 => cast_to!(Int128, i128),
                    Type::Uint16 => cast_to!(Uint16, u16),
                    Type::Uint32 => cast_to!(Uint32, u32),
                    Type::Uint64 => cast_to!(Uint64, u64),
                    Type::Uint128 => cast_to!(Uint128, u128),
                    _ => unreachable!(),
                }
            }
            Type::Float64 => match val {
                Value::Float64(f) => Ok(Value::Float64(f)),
                Value::Float32(f) => Ok(Value::Float64(f as f64)),
                Value::Int(n) => Ok(Value::Float64(n as f64)),
                Value::Uint(n) => Ok(Value::Float64(n as f64)),
                Value::Uint8(n) => Ok(Value::Float64(n as f64)),
                Value::Int8(n) => Ok(Value::Float64(n as f64)),
                Value::Int16(n) => Ok(Value::Float64(n as f64)),
                Value::Int32(n) => Ok(Value::Float64(n as f64)),
                Value::Int64(n) => Ok(Value::Float64(n as f64)),
                Value::Int128(n) => Ok(Value::Float64(n as f64)),
                Value::Uint16(n) => Ok(Value::Float64(n as f64)),
                Value::Uint32(n) => Ok(Value::Float64(n as f64)),
                Value::Uint64(n) => Ok(Value::Float64(n as f64)),
                Value::Uint128(n) => Ok(Value::Float64(n as f64)),
                Value::Str(s) => Ok(s.trim().parse::<f64>().map(Value::Float64).unwrap_or(Value::Nil)),
                _ => Ok(Value::Nil),
            },
            // `as float32` — bit-truncating narrowing from float64 (Rust `as f32`
            // semantics: silent precision loss, overflow saturates to infinity —
            // docs/float-width-types.md §4, no stricter checked narrowing invented here).
            Type::Float32 => match val {
                Value::Float32(f) => Ok(Value::Float32(f)),
                Value::Float64(f) => Ok(Value::Float32(f as f32)),
                Value::Int(n) => Ok(Value::Float32(n as f32)),
                Value::Uint(n) => Ok(Value::Float32(n as f32)),
                Value::Uint8(n) => Ok(Value::Float32(n as f32)),
                Value::Int8(n) => Ok(Value::Float32(n as f32)),
                Value::Int16(n) => Ok(Value::Float32(n as f32)),
                Value::Int32(n) => Ok(Value::Float32(n as f32)),
                Value::Int64(n) => Ok(Value::Float32(n as f32)),
                Value::Int128(n) => Ok(Value::Float32(n as f32)),
                Value::Uint16(n) => Ok(Value::Float32(n as f32)),
                Value::Uint32(n) => Ok(Value::Float32(n as f32)),
                Value::Uint64(n) => Ok(Value::Float32(n as f32)),
                Value::Uint128(n) => Ok(Value::Float32(n as f32)),
                Value::Str(s) => Ok(s.trim().parse::<f32>().map(Value::Float32).unwrap_or(Value::Nil)),
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

    /// Matches `pattern` against `value`, collecting bound names into `bindings`.
    ///
    /// `mut_names` collects the subset of those names that are bound directly
    /// to an enum variant field declared `mut Type` (see docs/book.md
    /// for the modifier itself) — the caller should `mark_content_mutable` each
    /// one on the arm's child env once bound, so `def` methods can be called
    /// through them (mirrors how `let`/param binding already does this for a
    /// `mut`-qualified type). Struct destructuring (`Point(x, y)`) never adds to
    /// `mut_names` — struct fields don't parse `mut Type` yet, only enum variant
    /// fields do (an accidental side effect of the generic `mut`-prefix type
    /// parser, not yet a deliberately supported struct feature).
    pub(crate) fn match_pattern(&self, pattern: &Pattern, value: &Value, bindings: &mut HashMap<String, Value>, mut_names: &mut HashSet<String>) -> bool {
        match pattern {
            Pattern::Wildcard => true,
            Pattern::Bind(name) => {
                bindings.insert(name.clone(), value.clone());
                true
            }
            Pattern::Lit(lit) => match (lit, value) {
                (LitPattern::Int(n), Value::Int(v)) => n == v,
                // For NaN: treat NaN == NaN as true (pattern matching, not arithmetic).
                (LitPattern::Float(f), Value::Float64(v)) => (f.is_nan() && v.is_nan()) || f == v,
                (LitPattern::Float(f), Value::Float32(v)) => (f.is_nan() && v.is_nan()) || *f == *v as f64,
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
                            self.match_pattern(inner, &fields[0], bindings, mut_names)
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
                    _ => self.match_pattern(inner, value, bindings, mut_names),
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
                            self.match_pattern(&sub_pats[0], value, bindings, mut_names)
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
                        // Declared field types for this variant, if the enum is known —
                        // used only to flag `mut`-qualified fields bound bare (`Pattern::Bind`)
                        // directly at this level into `mut_names`.
                        let field_decls = self.enums.get(type_name.as_str())
                            .and_then(|decl| decl.variants.iter().find(|v| &v.name == variant))
                            .map(|v| &v.fields);
                        for (i, (pat, field_val)) in sub_pats.iter().zip(fields.iter()).enumerate() {
                            if let (Pattern::Bind(bname), Some(fields_decl)) = (pat, field_decls) {
                                if fields_decl.get(i).map(|f| f.ty.grants_mut()).unwrap_or(false) {
                                    mut_names.insert(bname.clone());
                                }
                            }
                            if !self.match_pattern(pat, field_val, bindings, mut_names) {
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
                            if !self.match_pattern(pat, field_val, bindings, mut_names) {
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
                    if !self.match_pattern(pat, elem, bindings, mut_names) {
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
            Type::Int | Type::Uint | Type::Uint8 | Type::Float32 | Type::Float64 | Type::Bool => Ok(()),
            Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128 => Ok(()),
            // Bare String without qualifier: requires explicit qualification.
            Type::Str => Err(err("use 'string' instead of bare 'String' (which has no ownership qualifier)", line)),

            // Type parameters — resolved later, skip
            Type::TypeParam(_) => Ok(()),

            // Explicitly qualified — ownership is stated; no need to recurse into inner
            Type::Qualified(_, _) => Ok(()),

            // Bare named type → stack-owned (the default in boring).
            // `Dog d` = `Dog` on the stack, same as Rust's default.
            // Use `Dog&` for a borrow, `Dog'owned` for Box<Dog>.
            Type::Named(_) => Ok(()),

            // Optional — check the wrapped type
            Type::Optional(inner) => self.check_resolved_qualifier(inner, line),

            // Collection element / key / value types
            Type::Array(elem) | Type::ArrayN(elem, _) | Type::ArrayNExpr(elem, _) | Type::Set(elem) => self.check_resolved_qualifier(elem, line),
            Type::LabeledArray(elem, _) => self.check_resolved_qualifier(elem, line),
            Type::ConstInt(_) => Ok(()),
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

            // `mut Type` — a Boring-only permission, no ownership qualifier of its
            // own; check the wrapped type.
            Type::Mut(inner) => self.check_resolved_qualifier(inner, line),
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
            Value::Int(n)  => Ok(n),
            Value::Uint(n) => Ok(n as i64),
            Value::Uint8(n) => Ok(n as i64),
            Value::Int8(n) => Ok(n as i64),
            Value::Int16(n) => Ok(n as i64),
            Value::Int32(n) => Ok(n as i64),
            Value::Int64(n) => Ok(n),
            Value::Int128(n) => Ok(n as i64),
            Value::Uint16(n) => Ok(n as i64),
            Value::Uint32(n) => Ok(n as i64),
            Value::Uint64(n) => Ok(n as i64),
            Value::Uint128(n) => Ok(n as i64),
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
            Value::Uint8(_) => Type::Uint8,
            Value::Int8(_)  => Type::Int8,
            Value::Int16(_) => Type::Int16,
            Value::Int32(_) => Type::Int32,
            Value::Int64(_) => Type::Int64,
            Value::Int128(_) => Type::Int128,
            Value::Uint16(_) => Type::Uint16,
            Value::Uint32(_) => Type::Uint32,
            Value::Uint64(_) => Type::Uint64,
            Value::Uint128(_) => Type::Uint128,
            Value::Float32(_) => Type::Float32,
            Value::Float64(_) => Type::Float64,
            Value::Str(_)   => Type::Str,
            Value::Bool(_)  => Type::Bool,
            Value::Nil      => Type::Nil,
            Value::Void     => Type::Void,
            Value::Array(elems) => {
                let elem_ty = elems.first().map(Self::type_of_value).unwrap_or(Type::Nil);
                Type::Array(Box::new(elem_ty))
            }
            Value::Tuple(elems) => Type::Tuple(elems.iter().map(Self::type_of_value).collect()),
            Value::Dict(pairs) => {
                let k = pairs.first().map(|(k, _)| Self::type_of_value(k)).unwrap_or(Type::Nil);
                let v = pairs.first().map(|(_, v)| Self::type_of_value(v)).unwrap_or(Type::Nil);
                Type::Dict(Box::new(k), Box::new(v))
            }
            Value::Set(elems) => {
                let elem_ty = elems.first().map(Self::type_of_value).unwrap_or(Type::Nil);
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
                Type::Int | Type::Uint | Type::Uint8 | Type::Float64 | Type::Str | Type::Bool => continue,
                Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                    | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128 => continue,
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
                    if decl.protocols.iter().any(|p| p == trait_name) || decl.methods.iter().any(|m| m.qualifier.as_deref() == Some(trait_name.as_str())) { true }
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
                    if protocols.iter().any(|p| p == trait_name) || methods.iter().any(|m| m.qualifier.as_deref() == Some(trait_name.as_str())) { true }
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
                        Ok(Value::Array(lines.into()))
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
                        Ok(Value::Array(entries.into()))
                    }
                    Err(e) => fs_err!(e),
                }
            }
            "readBytes" => {
                let path = str_arg!(0);
                match std::fs::read(&path) {
                    Ok(bytes) => {
                        let arr = bytes.iter()
                            .map(|&b| Value::Uint8(b))
                            .collect::<Vec<_>>();
                        Ok(Value::Array(arr.into()))
                    }
                    Err(e) => fs_err!(e),
                }
            }
            "writeBytes" => {
                let path = str_arg!(0);
                let bytes = match args.get(1) {
                    Some(Value::Array(arr)) => {
                        let mut out = Vec::with_capacity(arr.len());
                        for v in arr.iter() {
                            match v {
                                Value::Uint8(n) => out.push(*n),
                                other => return Err(err(format!("fs.writeBytes: expected [uint8] elements, found {}", other.type_name()), line)),
                            }
                        }
                        out
                    }
                    _ => return Err(err("fs.writeBytes: expected [uint8] as second argument", line)),
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

