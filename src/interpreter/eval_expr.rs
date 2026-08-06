use super::*;
use std::collections::{HashMap, VecDeque};
use std::cell::RefCell;
use std::rc::Rc;

// ─── Generic numeric promotion ───────────────────────────────────────────────
//
// Boring has 12 numeric Value kinds (Int/Uint/Uint8 plus the 9 fixed-width types).
// Same-kind arithmetic is still hand-written per operator (native wrapping ops, same
// as the original Int/Uint/Uint8 arms). This section generalizes the OTHER supported
// case — mixing a fixed-width kind with the flexible bare `Int`/`Uint` literal kind
// (e.g. `some_uint32_var + 1`) — instead of hand-writing the full pairwise matrix.
// Mixing two *distinct* explicit fixed-width kinds directly (`uint16_val + int32_val`)
// is intentionally unsupported (falls through to the existing "cannot add X and Y"
// error, requiring an explicit cast) — this mirrors Rust's own refusal to implicitly
// coerce between distinct integer types.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NumKind { Int, Uint, Uint8, Int8, Int16, Int32, Int64, Int128, Uint16, Uint32, Uint64, Uint128 }

impl NumKind {
    fn of(v: &Value) -> Option<NumKind> {
        Some(match v {
            Value::Int(_) => NumKind::Int,
            Value::Uint(_) => NumKind::Uint,
            Value::Uint8(_) => NumKind::Uint8,
            Value::Int8(_) => NumKind::Int8,
            Value::Int16(_) => NumKind::Int16,
            Value::Int32(_) => NumKind::Int32,
            Value::Int64(_) => NumKind::Int64,
            Value::Int128(_) => NumKind::Int128,
            Value::Uint16(_) => NumKind::Uint16,
            Value::Uint32(_) => NumKind::Uint32,
            Value::Uint64(_) => NumKind::Uint64,
            Value::Uint128(_) => NumKind::Uint128,
            _ => return None,
        })
    }

    fn bits(self) -> u32 {
        use NumKind::*;
        match self {
            Int8 | Uint8 => 8,
            Int16 | Uint16 => 16,
            Int32 | Uint32 => 32,
            Int | Uint | Int64 | Uint64 => 64,
            Int128 | Uint128 => 128,
        }
    }

    fn signed(self) -> bool {
        use NumKind::*;
        matches!(self, Int | Int8 | Int16 | Int32 | Int64 | Int128)
    }

    fn is_bare(self) -> bool { matches!(self, NumKind::Int | NumKind::Uint) }

    fn name(self) -> &'static str {
        match self {
            NumKind::Int => "int", NumKind::Uint => "uint", NumKind::Uint8 => "uint8",
            NumKind::Int8 => "int8", NumKind::Int16 => "int16", NumKind::Int32 => "int32",
            NumKind::Int64 => "int64", NumKind::Int128 => "int128",
            NumKind::Uint16 => "uint16", NumKind::Uint32 => "uint32",
            NumKind::Uint64 => "uint64", NumKind::Uint128 => "uint128",
        }
    }

    /// Result kind when mixing two *different* kinds where at least one is the bare
    /// `Int`/`Uint`. Returns `None` when neither is bare (unsupported direct mix).
    ///
    /// Rule: the wider kind wins. On a tied width with mixed signs: for every op
    /// except `Sub`, unsigned wins (matches the original `Int`+`Uint` precedent,
    /// symmetric regardless of operand order). `Sub`'s existing precedent is
    /// order-dependent (`Uint - Int` stays Uint, `Int - Uint` stays Int) — generalized
    /// here as "the LHS's kind wins on a tied-width sign mismatch".
    fn promote(ka: NumKind, kb: NumKind, is_sub: bool) -> Option<NumKind> {
        if ka == kb { return None; }
        if !ka.is_bare() && !kb.is_bare() { return None; }
        if ka.bits() != kb.bits() {
            return Some(if ka.bits() > kb.bits() { ka } else { kb });
        }
        if ka.signed() == kb.signed() {
            // Tied width, same sign, different kind label (bare vs. same-width specific
            // type, e.g. `Int` vs `Int64`) — prefer the more specific, non-bare kind.
            return Some(if ka.is_bare() { kb } else { ka });
        }
        if is_sub {
            Some(ka)
        } else {
            Some(if ka.signed() { kb } else { ka }) // unsigned wins
        }
    }
}

/// Widen any numeric Value's payload into i128 staging space. Returns `None` for
/// non-numeric values and for `Uint128` values above `i128::MAX` — that ultra-wide
/// corner case is intentionally not supported for mixed-kind arithmetic (same-kind
/// `Uint128 op Uint128` still works fully via native `u128` ops elsewhere).
fn value_as_i128(v: &Value) -> Option<i128> {
    Some(match v {
        Value::Int(n) => *n as i128,
        Value::Uint(n) => *n as i128,
        Value::Uint8(n) => *n as i128,
        Value::Int8(n) => *n as i128,
        Value::Int16(n) => *n as i128,
        Value::Int32(n) => *n as i128,
        Value::Int64(n) => *n as i128,
        Value::Int128(n) => *n,
        Value::Uint16(n) => *n as i128,
        Value::Uint32(n) => *n as i128,
        Value::Uint64(n) => *n as i128,
        Value::Uint128(n) => i128::try_from(*n).ok()?,
        _ => return None,
    })
}

/// Narrow an i128 staging value back into `kind`'s native Value (wrapping, matching
/// the `wrapping_*` semantics used throughout the same-kind arithmetic arms).
fn i128_to_kind(kind: NumKind, x: i128) -> Value {
    match kind {
        NumKind::Int => Value::Int(x as i64),
        NumKind::Uint => Value::Uint(x as u64),
        NumKind::Uint8 => Value::Uint8(x as u8),
        NumKind::Int8 => Value::Int8(x as i8),
        NumKind::Int16 => Value::Int16(x as i16),
        NumKind::Int32 => Value::Int32(x as i32),
        NumKind::Int64 => Value::Int64(x as i64),
        NumKind::Int128 => Value::Int128(x),
        NumKind::Uint16 => Value::Uint16(x as u16),
        NumKind::Uint32 => Value::Uint32(x as u32),
        NumKind::Uint64 => Value::Uint64(x as u64),
        NumKind::Uint128 => Value::Uint128(x as u128),
    }
}

/// Generic fallback for a binary numeric op between two *different* numeric kinds,
/// used from the catch-all arm of each arithmetic/bitwise operator's match — after
/// the hand-written same-kind and legacy `Int`/`Uint`/`Uint8` arms have already had
/// first shot. Returns `None` when the pair isn't a supported mixed combination
/// (the caller falls through to the existing "cannot <op> X and Y" error).
fn eval_numeric_mixed(l: &Value, r: &Value, op: &BinOp, line: usize, rcol: usize, rlen: usize) -> Option<Result<Value, Signal>> {
    let ka = NumKind::of(l)?;
    let kb = NumKind::of(r)?;
    let is_sub = matches!(op, BinOp::Sub);
    let result_kind = NumKind::promote(ka, kb, is_sub)?;
    let lv = value_as_i128(l)?;
    let rv = value_as_i128(r)?;

    if !result_kind.signed() {
        // A negative LHS can never land in an unsigned result.
        if ka.signed() && lv < 0 {
            return Some(Err(err_span(format!("cannot combine negative {} with unsigned type", l.type_name()), line, rcol, rlen)));
        }
        // A negative RHS is allowed only for Sub (subtracting a negative == adding),
        // mirroring the original Uint - Int special case.
        if kb.signed() && rv < 0 && !is_sub {
            return Some(Err(err_span(format!("cannot combine negative {} with unsigned type", r.type_name()), line, rcol, rlen)));
        }
    }

    let result = match op {
        BinOp::Add => lv.wrapping_add(rv),
        BinOp::Sub => {
            if !result_kind.signed() && rv >= 0 && rv > lv {
                return Some(Err(err_span(format!("{} subtraction underflow", result_kind.name()), line, rcol, rlen)));
            }
            lv.wrapping_sub(rv)
        }
        BinOp::Mul => lv.wrapping_mul(rv),
        BinOp::Div => {
            if rv == 0 { return Some(Err(err_span("division by zero", line, rcol, rlen))); }
            lv.wrapping_div(rv)
        }
        BinOp::Rem => {
            if rv == 0 { return Some(Err(err_span("remainder by zero", line, rcol, rlen))); }
            lv.wrapping_rem(rv)
        }
        BinOp::BitAnd => lv & rv,
        BinOp::BitOr  => lv | rv,
        BinOp::BitXor => lv ^ rv,
        BinOp::Shl => {
            if rv < 0 { return Some(Err(err_span("shift amount cannot be negative", line, rcol, rlen))); }
            lv.wrapping_shl(rv as u32)
        }
        BinOp::Shr => {
            if rv < 0 { return Some(Err(err_span("shift amount cannot be negative", line, rcol, rlen))); }
            lv.wrapping_shr(rv as u32)
        }
        _ => return None,
    };
    Some(Ok(i128_to_kind(result_kind, result)))
}

/// Parse `Screen(Dimension(w,h), title="...")` or `Screen(w, h, title="...")` arguments.
fn parse_screen_args(args: &[Value]) -> (u64, u64, String) {
    let mut width: u64 = 800;
    let mut height: u64 = 600;
    let mut title = "Boring".to_string();

    let mut positional_idx = 0;
    for arg in args {
        match arg {
            Value::Labeled { label, value } if label == "title" => {
                if let Value::Str(s) = value.as_ref() {
                    title = s.to_string();
                }
            }
            Value::Object(obj) if obj.borrow().type_name == "Dimension" => {
                let obj = obj.borrow();
                for (k, v) in &obj.fields {
                    match (k.as_str(), v) {
                        ("width",  Value::Uint(n)) => width  = *n,
                        ("height", Value::Uint(n)) => height = *n,
                        _ => {}
                    }
                }
            }
            Value::Uint(n) => {
                if positional_idx == 0 { width  = *n; }
                else                   { height = *n; }
                positional_idx += 1;
            }
            Value::Int(n) => {
                if positional_idx == 0 { width  = *n as u64; }
                else                   { height = *n as u64; }
                positional_idx += 1;
            }
            _ => {}
        }
    }
    (width, height, title)
}

impl Interpreter {
    /// Fast path for `var_name.mutatingArrayMethod(args)` on a plain local variable.
    ///
    /// The generic `MethodCall` handling reads the receiver via `eval_expr` (which
    /// clones the `Rc<Vec<Value>>` out of the env slot while the slot's own copy
    /// stays alive too), so by the time the mutation runs, at least two owners of
    /// the same `Vec` are alive and copy-on-write (`Value::rc_vec_into_owned`)
    /// always has to deep-clone — an O(n) cost on every `.push()` in a loop, i.e.
    /// O(n^2) overall. Taking the value out of the slot instead (leaving a `Nil`
    /// placeholder) drops that to one owner in the common unaliased case, making
    /// `push` etc. O(1) amortized instead of O(n) per call.
    ///
    /// Restricted to methods that can never fail after taking ownership — `pop`,
    /// `removeAt`, and `sortBy` can error out (empty array / bad index / a
    /// throwing closure) after ownership is already taken, and unlike the generic
    /// path (which only ever operates on a *clone*, leaving the env's own copy
    /// untouched) there would be no original left to restore. Those stay on the
    /// slower, always-safe generic path below.
    ///
    /// Returns `None` if the fast path doesn't apply (wrong method, or the
    /// variable doesn't currently hold an array) — the caller falls through to
    /// the generic path unchanged. `#[inline(never)]` so this stays its own stack
    /// frame rather than inflating `eval_expr`'s (a big, deeply-recursive match)
    /// on every call.
    #[inline(never)]
    pub(crate) fn try_fast_mutating_array_call(
        &mut self,
        name: &str,
        method: &str,
        args: &[Arg],
        env: &EnvRef,
        line: usize,
    ) -> Option<Eval> {
        const FAST_MUTATING_ARRAY_METHODS: &[&str] =
            &["push", "append", "insert", "remove", "sort", "reverse"];
        if !FAST_MUTATING_ARRAY_METHODS.contains(&method)
            || !matches!(env.borrow().get(name), Some(Value::Array(_)))
        {
            return None;
        }
        Some((|| {
            let arg_vals = self.eval_args(args, Rc::clone(env))?;
            let taken = env.borrow_mut().take(name).ok_or_else(|| {
                err(format!("internal error: variable '{}' disappeared", name), line)
            })?;
            let mut modified_self: Option<Value> = None;
            let result = self.call_method(taken, method, arg_vals, line, &mut modified_self)?;
            env.borrow_mut().force_set(name, modified_self.unwrap_or(Value::Nil));
            Ok(result)
        })())
    }

    /// `arr[i].min/max/swap/cas(...)` on an indexed element — see the call
    /// site's own doc comment for the full rationale. Returns `None` (not
    /// this method) for any other method name, so the caller falls through
    /// to ordinary method dispatch.
    fn try_index_atomic_method_call(
        &mut self,
        obj_expr: &Expr,
        method: &str,
        args: &[Arg],
        env: &EnvRef,
        line: usize,
    ) -> Option<Eval> {
        if !matches!(method, "min" | "max" | "swap" | "cas") {
            return None;
        }
        Some((|| {
            let old = self.eval_expr(obj_expr, Rc::clone(env))?;
            let new_val = match method {
                "min" => {
                    let v = self.eval_expr(&args[0].value, Rc::clone(env))?;
                    if self.compare_values(old.clone(), v.clone(), |o| o != std::cmp::Ordering::Greater, line, 0)? == Value::Bool(true) {
                        old.clone()
                    } else {
                        v
                    }
                }
                "max" => {
                    let v = self.eval_expr(&args[0].value, Rc::clone(env))?;
                    if self.compare_values(old.clone(), v.clone(), |o| o != std::cmp::Ordering::Less, line, 0)? == Value::Bool(true) {
                        old.clone()
                    } else {
                        v
                    }
                }
                "swap" => self.eval_expr(&args[0].value, Rc::clone(env))?,
                "cas" => {
                    let expected = self.eval_expr(&args[0].value, Rc::clone(env))?;
                    let new = self.eval_expr(&args[1].value, Rc::clone(env))?;
                    if Self::values_equal(&old, &expected) { new } else { old.clone() }
                }
                _ => unreachable!(),
            };
            self.assign(obj_expr, new_val, Rc::clone(env), line)?;
            Ok(old)
        })())
    }

    /// `callee(args)` — built-in namespace calls (`channel`/`timeout`/`json`/`Dimension`/
    /// `Screen`/`GPU`/print-family), the implicit-`self` fallback (`foo(args)` inside a
    /// method resolves to `self.foo(args)` when `foo` isn't otherwise in scope), the
    /// `k(block = N)` kernel-launch shorthand, and owned/`var`-param bookkeeping around
    /// an ordinary call.
    fn eval_expr_call(&mut self, callee_expr: &Expr, args: &[Arg], env: EnvRef, line: usize) -> Eval {
        // channel/oneshot/broadcast/watch without type args: return (Sender, Receiver) pair
        if let ExprKind::Var(name) = &callee_expr.kind {
            if matches!(name.as_str(), "channel" | "oneshot" | "broadcast" | "watch") {
                let buf = Rc::new(RefCell::new(VecDeque::new()));
                let closed = Rc::new(RefCell::new(false));
                let sender = Value::Channel { buf: Rc::clone(&buf), closed: Rc::clone(&closed), is_sender: true };
                let receiver = Value::Channel { buf, closed, is_sender: false };
                return Ok(Value::Tuple(vec![sender, receiver]));
            }
            // timeout(dur, fut_or_callable) — interpreter: skip duration, evaluate the future.
            // Two forms:
            //   timeout(dur, task f(args))  — second arg is already a Future expression
            //   timeout(dur, f)             — second arg is a Callable<T>: call it to get Future
            if name.as_str() == "timeout" {
                if let Some(fut_arg) = args.get(1) {
                    let val = self.eval_expr(&fut_arg.value, Rc::clone(&env))?;
                    // If the second arg is a Fn/Closure (Callable<T>), call it with no args.
                    return match val {
                        v @ (Value::Fn { .. } | Value::Closure { .. } | Value::NativeFn { .. }) => {
                            let result = self.call_value(v, vec![], fut_arg.value.line, false)?;
                            // Unwrap Future if the callable returned one
                            match result {
                                Value::Future(inner) => Ok(*inner),
                                other => Ok(other),
                            }
                        }
                        Value::Future(inner) => Ok(*inner),
                        other => Ok(other),
                    };
                }
                return Ok(Value::Nil);
            }
            // json(v) — interpreter stub: convert value to its debug string representation
            if name.as_str() == "json" {
                if let Some(arg) = args.first() {
                    let v = self.eval_expr(&arg.value, Rc::clone(&env))?;
                    return Ok(Value::Str(format!("{:?}", v)));
                }
                return Ok(Value::Str("null".into()));
            }
            // Dimension(w, h) — built-in size descriptor.
            // Stored as an Object with fields `width` and `height`.
            if name.as_str() == "Dimension" {
                let arg_vals = self.eval_args(args, Rc::clone(&env))?;
                let (w, h) = match arg_vals.as_slice() {
                    [Value::Uint(w), Value::Uint(h)] => (*w, *h),
                    [Value::Int(w), Value::Int(h)] => (*w as u64, *h as u64),
                    [Value::Uint(w)] => (*w, *w),
                    _ => (0, 0),
                };
                let obj = crate::interpreter::ObjectInner {
                    type_name: "Dimension".into(),
                    fields: vec![
                        ("width".into(),  Value::Uint(w)),
                        ("height".into(), Value::Uint(h)),
                    ],
                };
                return Ok(Value::Object(Rc::new(RefCell::new(obj))));
            }
            // Screen(Dimension | w, h, title = ...) — built-in window (simulation mode).
            if name.as_str() == "Screen" {
                let arg_vals = self.eval_args(args, Rc::clone(&env))?;
                let (w, h, title) = parse_screen_args(&arg_vals);
                return Ok(Value::Screen {
                    width:   Rc::new(RefCell::new(w)),
                    height:  Rc::new(RefCell::new(h)),
                    title,
                    frame:   Rc::new(RefCell::new(0)),
                    resized: Rc::new(RefCell::new(false)),
                    keys:    Rc::new(RefCell::new(vec![])),
                    pixels:  Rc::new(RefCell::new(vec![])),
                });
            }
            // GPU(n) — built-in GPU device handle (simulation mode).
            if name.as_str() == "GPU" {
                let idx = match args.first() {
                    Some(a) => match self.eval_expr(&a.value, Rc::clone(&env))? {
                        Value::Int(n) => n as usize,
                        Value::Uint(n) => n as usize,
                        _ => 0,
                    },
                    None => 0,
                };
                return Ok(Value::GpuDevice(idx));
            }
            // print / write / log-level — use `as string:` conversions for Object args
            if matches!(name.as_str(), "print" | "write" | "error" | "warn" | "info" | "debug" | "trace") {
                let arg_vals = self.eval_args(args, Rc::clone(&env))?;
                return self.call_display_builtin(name, &arg_vals, line);
            }
        }
        // Implicit self method call: `foo(args)` inside a struct method —
        // if `foo` isn't in scope but `self` is, try `self.foo(args)`.
        if let ExprKind::Var(name) = &callee_expr.kind {
            let not_in_scope = env.borrow().get(name).is_none();
            if not_in_scope {
                let self_opt = env.borrow().get("self"); // borrow released after this line
                if let Some(self_val) = self_opt {
                    let arg_vals = self.eval_args(args, Rc::clone(&env))?;
                    let mut modified_self: Option<Value> = None;
                    match self.call_method(self_val, name, arg_vals, line, &mut modified_self) {
                        Ok(result) => {
                            if let Some(new_self) = modified_self {
                                env.borrow_mut().force_set("self", new_self);
                            }
                            return Ok(result);
                        }
                        Err(e) => {
                            return Err(e);
                        }
                    }
                }
            }
        }
        let callee = self.eval_expr(callee_expr, Rc::clone(&env))?;
        // `k(block = N)` short-hand — if the callee is a kernel Object and the
        // arguments contain a `block =` labeled arg, treat this as a kernel launch
        // and return a KernelHandle. The handle immediately returns `k` on `.wait`.
        if matches!(&callee, Value::Object(_)) {
            let type_name = callee.type_name();
            let is_kernel = self.global.borrow().get(&type_name)
                .map(|v| matches!(v, Value::KernelStruct { .. }))
                .unwrap_or(false);
            let has_block = args.iter().any(|a| a.label.as_deref() == Some("block"));
            let has_grid  = args.iter().any(|a| a.label.as_deref() == Some("grid"));
            if is_kernel && has_block && has_grid {
                // `grid =` is given explicitly alongside `block =` — dispatch
                // through `eval_kernel_launch`, which actually honors `grid`
                // (parses int/tuple, up to 3D). The shorthand path below
                // (`eval_kernel_launch_with_val`) NEVER looks at a `grid=`
                // argument at all, even when one is present — it always
                // infers grid from the longest array-typed field divided by
                // block size. That inference silently produces the wrong
                // grid whenever the largest field's length doesn't happen to
                // correspond to the intended dispatch shape (e.g. a `'global`
                // input array bigger than the output, common in tiled GEMM
                // kernels) — invisible in small single-block test kernels,
                // where both the explicit and the inferred grid are 1 anyway.
                let block_expr = args.iter().find(|a| a.label.as_deref() == Some("block")).map(|a| a.value.clone());
                let grid_expr  = args.iter().find(|a| a.label.as_deref() == Some("grid")).map(|a| a.value.clone());
                let config = crate::ast::KernelConfig {
                    block: block_expr, grid: grid_expr, after: None, priority: None, line, col: 0,
                };
                return self.eval_kernel_launch(&config, callee_expr, env);
            }
            if is_kernel && has_block {
                // Build a synthetic KernelConfig from the labeled args.
                let block_arg = args.iter().find(|a| a.label.as_deref() == Some("block"))
                    .map(|a| self.eval_expr(&a.value, Rc::clone(&env)));
                let block_val = match block_arg {
                    Some(Ok(v)) => v,
                    _ => Value::Int(1),
                };
                let after_arg = args.iter().find(|a| a.label.as_deref() == Some("after"))
                    .map(|a| self.eval_expr(&a.value, Rc::clone(&env)));
                let _after_val = after_arg.and_then(|r| r.ok());
                let config = crate::ast::KernelConfig {
                    block: Some(Expr { kind: crate::ast::ExprKind::Nil, line, col: 0, len: 0 }),
                    grid: None, after: None, priority: None, line, col: 0,
                };
                return self.eval_kernel_launch_with_val(config, callee, block_val, line, &env);
            }
        }
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
        // Write back mutated `var` params to their caller variables.
        if let Value::Fn { ref decl, .. } = callee {
            for (param, arg) in decl.params.iter().zip(args.iter()) {
                if param.mutable {
                    if let ExprKind::Var(caller_name) = &arg.value.kind {
                        if let Some(new_val) = self.last_var_params.get(&param.name).cloned() {
                            env.borrow_mut().force_set(caller_name, new_val);
                        }
                    }
                }
            }
        }
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

    /// `obj.method(args)` — the `fs`/`GPU.all()` builtin namespaces, type-level calls
    /// (`Counter.zero()`), the fast-path mutating-array-method shortcut, GPU-device
    /// property mocks, the immutable-`let`-binding mutating-method diagnostic, and an
    /// ordinary struct method call with its modified-`self`/owned-param write-back.
    fn eval_expr_method_call(&mut self, obj_expr: &Expr, method: &str, args: &[Arg], env: EnvRef, line: usize) -> Eval {
        // `gpu.warp.sync()` / `gpu.warp.shuffle_down/up/xor/shuffle(...)` — matched
        // purely on the receiver's AST shape (`gpu.warp` is never evaluated as a
        // real value; only `.size`/`.lane` field access goes through the `GpuWarp`
        // object `run_one_kernel_thread` injects), same style as the `fs` namespace
        // check just below.
        if let ExprKind::Field(inner, ns) = &obj_expr.kind {
            if ns == "warp" {
                if let ExprKind::Var(g) = &inner.kind {
                    if g == "gpu" {
                        return self.eval_gpu_warp_method(method, args, env, line);
                    }
                }
            }
        }
        // `arr[i].min/max/swap/cas(...)` — read-modify-write on an indexed
        // element, returning the OLD value. Matches the atomic
        // atomicMin/Max/Exch/CAS intrinsics the transpiler backends emit for
        // the identical call shape on an `'actor'global`/`'actor'unified`
        // kernel field, but the interpreter doesn't check the qualifier
        // here — same precedent as `+=`/`-=`/etc. on any array element,
        // which already works via ordinary compound-assign desugaring
        // (`Assign(target, BinOp(op, target, value))`) with no special-cased
        // atomicity anywhere in this single-logical-thread-per-kernel-instance
        // simulation. `self.assign` is the exact same write-back that
        // desugared compound-assign already relies on.
        if matches!(&obj_expr.kind, ExprKind::Index(..)) {
            if let Some(result) = self.try_index_atomic_method_call(obj_expr, method, args, &env, line) {
                return result;
            }
        }
        // Built-in `fs` module namespace — intercept before evaluating the receiver
        // so that `fs` does not need to be defined as a variable.
        if let ExprKind::Var(v) = &obj_expr.kind {
            if v == "fs" {
                let arg_vals = self.eval_args(args, Rc::clone(&env))?;
                return self.call_fs_method(method, arg_vals, line);
            }
            // GPU.all() — iterate all GPU devices (simulation: a single device).
            if v == "GPU" && method == "all" {
                return Ok(Value::Array(vec![Value::GpuDevice(0)].into()));
            }
        }

        // Type-level call: `Counter.zero()` or `Counter.set_count(v)`
        if let ExprKind::Var(type_name) = &obj_expr.kind {
            // Task.cancelled() — not supported in the interpreter (no cancellation
            // token), so always return false to keep code that uses it runnable.
            if type_name == "Task" && method == "cancelled" && args.is_empty() {
                return Ok(Value::Bool(false));
            }
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
        // Fast path: `var_name.mutatingArrayMethod(args)` on a plain local
        // variable. See `try_fast_mutating_array_call`'s doc comment. Kept in
        // its own #[inline(never)] function so this large `eval_expr` match
        // doesn't gain extra stack-frame size for every call (this recursive
        // function runs close to the debug-build stack limit in some existing
        // call chains — inlining this directly here previously overflowed it).
        if let ExprKind::Var(name) = &obj_expr.kind {
            if let Some(result) = self.try_fast_mutating_array_call(name, method, args, &env, line) {
                return result;
            }
        }
        let obj = self.eval_expr(obj_expr, Rc::clone(&env))?;
        // GPU device property methods — simulation mock values.
        if let Value::GpuDevice(idx) = &obj {
            let idx = *idx;
            let p = &self.gpu_profile;
            let (cc_major, cc_minor) = p.compute_capability;
            return Ok(match method {
                "name"              => Value::Str(format!("{} (sim {})", p.name, idx)),
                "totalMem"          => Value::Int(p.total_mem),
                "freeMem"           => Value::Int(p.total_mem),  // nothing else running
                "computeCapability" => Value::Array(vec![Value::Int(cc_major), Value::Int(cc_minor)].into()),
                "warpSize"          => Value::Int(p.warp_size),
                "maxThreads"        => Value::Int(p.max_threads),
                "maxSharedMem"      => Value::Int(p.max_shared_mem),
                "index"             => Value::Int(idx as i64),
                other => return Err(err(
                    format!("GPU has no property '{other}'"), line,
                )),
            });
        }
        // Enforce: mutating method cannot be called on an immutable (let) binding
        // Built-in non-mutating methods (e.g. `upgrade`) bypass this check.
        const BUILTIN_NON_MUTATING: &[&str] = &["upgrade", "clone"];
        if let ExprKind::Var(binding_name) = &obj_expr.kind {
            if !BUILTIN_NON_MUTATING.contains(&method) {
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
                    let is_interior_mutable = env.borrow().is_actor(binding_name);
                    if is_mutating && !env.borrow().is_mutable(binding_name) {
                        return Err(err(
                            format!("cannot call mutating method '{}' on let binding '{}'", method, binding_name),
                            line,
                        ));
                    }
                    if is_mutating && !is_interior_mutable && env.borrow().is_shared(binding_name) {
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
                ExprKind::Var(name) => {
                    let written = env.borrow_mut().force_set(name, new_obj.clone());
                    // If not found in scope, try to write to self field (implicit self pattern).
                    if !written {
                        if let Some(Value::Object(ref inner_rc)) = env.borrow().get("self").as_ref() {
                            if inner_rc.borrow().fields.iter().any(|(k, _)| k == name.as_str()) {
                                let self_expr = Expr { kind: ExprKind::Var("self".to_string()), line, col: 0, len: 0 };
                                let field_expr = Expr { kind: ExprKind::Field(Box::new(self_expr), name.clone()), line, col: 0, len: 0 };
                                let _ = self.assign(&field_expr, new_obj, Rc::clone(&env), line);
                            }
                        }
                    }
                }
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
                    decl.methods.iter().find(|m| m.name == *method).map(|fn_decl| fn_decl.params.iter().zip(args.iter()).filter_map(|(param, arg)| {
                            if param.owned {
                                if let ExprKind::Var(name) = &arg.value.kind {
                                    return Some(name.clone());
                                }
                            }
                            None
                        }).collect())
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

    /// `obj[idx]` — slice ranges (`a[M..N]`/`a[..N]`/`a[M..]`/`a[..]`, negative-index-aware,
    /// on `Array` or `Str`) and plain single-index access via `get_index`.
    fn eval_expr_index(&mut self, obj_expr: &Expr, idx_expr: &Expr, env: EnvRef, line: usize) -> Eval {
        // Slice: a[M..N], a[..N], a[M..], a[..]
        if let ExprKind::SliceRange { start, end, inclusive } = &idx_expr.kind {
            let obj = self.eval_expr(obj_expr, Rc::clone(&env))?;
            match obj {
                Value::Array(arr) => {
                    let len = arr.len() as i64;
                    let resolve = |v: i64| -> usize {
                        let i = if v < 0 { (len + v).max(0) } else { v.min(len) };
                        i as usize
                    };
                    let lo = match start.as_deref() {
                        Some(e) => {
                            let Value::Int(v) = self.eval_expr(e, Rc::clone(&env))? else {
                                return Err(err("slice start must be an integer", line));
                            };
                            resolve(v)
                        }
                        None => 0,
                    };
                    let hi = match end.as_deref() {
                        Some(e) => {
                            let Value::Int(v) = self.eval_expr(e, Rc::clone(&env))? else {
                                return Err(err("slice end must be an integer", line));
                            };
                            if *inclusive { (resolve(v) + 1).min(arr.len()) } else { resolve(v) }
                        }
                        None => arr.len(),
                    };
                    let slice = if lo >= arr.len() || lo >= hi {
                        vec![]
                    } else {
                        arr[lo..hi.min(arr.len())].to_vec()
                    };
                    return Ok(Value::Array(slice.into()));
                }
                Value::Str(s) => {
                    let chars: Vec<char> = s.chars().collect();
                    let len = chars.len() as i64;
                    let resolve = |v: i64| -> usize {
                        let i = if v < 0 { (len + v).max(0) } else { v.min(len) };
                        i as usize
                    };
                    let lo = match start.as_deref() {
                        Some(e) => {
                            let Value::Int(v) = self.eval_expr(e, Rc::clone(&env))? else {
                                return Err(err("slice start must be an integer", line));
                            };
                            resolve(v)
                        }
                        None => 0,
                    };
                    let hi = match end.as_deref() {
                        Some(e) => {
                            let Value::Int(v) = self.eval_expr(e, Rc::clone(&env))? else {
                                return Err(err("slice end must be an integer", line));
                            };
                            if *inclusive { (resolve(v) + 1).min(chars.len()) } else { resolve(v) }
                        }
                        None => chars.len(),
                    };
                    let sliced: String = if lo >= chars.len() || lo >= hi {
                        String::new()
                    } else {
                        chars[lo..hi.min(chars.len())].iter().collect()
                    };
                    return Ok(Value::Str(sliced));
                }
                other => return Err(err(
                    format!("slice index requires an array or string, got {}", other.type_name()),
                    line,
                )),
            }
        }
        let obj = self.eval_expr(obj_expr, Rc::clone(&env))?;
        let idx = self.eval_expr(idx_expr, Rc::clone(&env))?;
        self.get_index(obj, idx, line, idx_expr.col, idx_expr.len)
    }

    /// `try stmts... else stmts...` (multi-line try/else block). Runs the try body; on
    /// an `Err(...)` `Result` variant or a thrown `Signal::Exception`, runs the else body
    /// with `error` bound to the **original thrown value** (not its string form), so the
    /// else body can `match error:` on a typed enum. Unwraps a bare `Ok(v)` on success.
    fn eval_expr_try_else_block(&mut self, try_stmts: &[Stmt], else_stmts: &[Stmt], env: EnvRef) -> Eval {
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

    /// `callee<T>(args)` — type args are erased at runtime, so this is a regular call
    /// plus the handful of builtins that need type info at the *call syntax* level
    /// (`channel<T>`/`oneshot<T>`/etc., `timeout<T>`, `fromJson<T>`), mirroring the
    /// untyped forms `eval_expr_call` already handles.
    fn eval_expr_generic_call(&mut self, callee: &Expr, args: &[Arg], env: EnvRef, line: usize) -> Eval {
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
                    let val = self.eval_expr(&fut_arg.value, Rc::clone(&env))?;
                    return match val {
                        v @ (Value::Fn { .. } | Value::Closure { .. } | Value::NativeFn { .. }) => {
                            let result = self.call_value(v, vec![], fut_arg.value.line, false)?;
                            match result {
                                Value::Future(inner) => Ok(*inner),
                                other => Ok(other),
                            }
                        }
                        Value::Future(inner) => Ok(*inner),
                        other => Ok(other),
                    };
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

    pub fn eval_expr(&mut self, expr: &Expr, env: EnvRef) -> Eval {
        let line = expr.line;
        let col = expr.col;
        let len = expr.len;
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
                // `'sync` fields: materialize a fresh snapshot from block-shared
                // storage on every read (not just indexed reads — `for v in tile:`,
                // whole-array copies, `.length`, etc. all read the bare Var) so any
                // consumer sees writes other threads in the block made before the
                // last barrier. See `Interpreter::sync_fields`'s doc comment.
                if let Some(shared) = self.sync_fields.get(name) {
                    let arr = shared.lock().unwrap();
                    let vals: Vec<Value> = arr.iter()
                        .map(|tv| super::eval_gpu::from_thread_value(tv.clone(), &env))
                        .collect();
                    return Ok(Value::Array(vals.into()));
                }
                if let Some(val) = env.borrow().get(name) {
                    if matches!(val, Value::Uninitialized) {
                        return Err(err(format!("variable '{}' used before being assigned", name), line));
                    }
                    if let Value::Moved(src) = &val {
                        return Err(err(format!("use of moved value '{}': the value was moved and is no longer accessible — use .clone() to make a copy", src), line));
                    }
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
                if let Some(Value::Object(inner_rc)) = env.borrow().get("self").as_ref() {
                    if let Some((_, field_val)) = inner_rc.borrow().fields.iter().find(|(k, _)| k == name) {
                        return Ok(field_val.clone());
                    }
                }
                let candidates = env.borrow().all_names();
                let msg = match closest_name(name, &candidates) {
                    Some(s) => format!("undefined variable '{}' — did you mean '{}'?", name, s),
                    None    => format!("undefined variable '{}'", name),
                };
                Err(err_span(msg, line, col, len))
            }

            ExprKind::BinOp(op, lhs, rhs) => {
                self.eval_binop(op, lhs, rhs, env, line, col)
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
                // Simultaneous multi-target: `a, b = b, a` — evaluate all RHS first.
                if let ExprKind::Tuple(targets) = &target.kind {
                    let rhs_vals: Vec<Value> = if let ExprKind::Tuple(rhs_list) = &value.kind {
                        rhs_list.iter().map(|e| self.eval_expr(e, Rc::clone(&env))).collect::<Result<_, _>>()?
                    } else {
                        let v = self.eval_expr(value, Rc::clone(&env))?;
                        match v { Value::Tuple(vs) => vs, single => vec![single] }
                    };
                    for (t, v) in targets.iter().zip(rhs_vals) {
                        self.assign(t, v, Rc::clone(&env), line)?;
                    }
                    return Ok(Value::Nil);
                }
                // Guard: prevent `=` on a lazy binding — use `?=` instead.
                if let ExprKind::Var(name) = &target.kind {
                    if env.borrow().is_lazy(name) {
                        return Err(err(
                            format!("cannot use '=' on lazy binding '{}', use '?=' to initialize it", name),
                            line,
                        ));
                    }
                }
                let val = self.eval_expr(value, Rc::clone(&env))?;
                self.assign(target, val.clone(), Rc::clone(&env), line)?;
                Ok(val)
            }

            // `w ?= expr` — write-once / nil-coalescing assign.
            // For lazy vars: initialize if not yet set; subsequent calls are no-ops.
            // For optional vars: assign if currently nil.
            ExprKind::QuestionAssign(target, rhs) => {
                // If target is a lazy var: initialize it once (regardless of current value).
                if let ExprKind::Var(name) = &target.kind {
                    if env.borrow().is_lazy(name) {
                        let val = self.eval_expr(rhs, Rc::clone(&env))?;
                        env.borrow_mut().set(name, val.clone())
                            .map_err(|_| err(format!("lazy binding '{}' has already been initialized", name), line))?;
                        return Ok(val);
                    }
                }
                // For non-lazy: null-coalescing — assign rhs only if current value is Nil.
                let current = self.eval_expr(target, Rc::clone(&env))?;
                if matches!(current, Value::Nil) {
                    let val = self.eval_expr(rhs, Rc::clone(&env))?;
                    self.assign(target, val.clone(), Rc::clone(&env), line)?;
                    Ok(val)
                } else {
                    Ok(current)
                }
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

            ExprKind::Index(obj_expr, idx_expr) => self.eval_expr_index(obj_expr, idx_expr, env, line),

            ExprKind::Call(callee_expr, args) => self.eval_expr_call(callee_expr, args, env, line),

            ExprKind::MethodCall(obj_expr, method, args) => self.eval_expr_method_call(obj_expr, method, args, env, line),

            ExprKind::New { ctor, .. } => {
                // Arena placement has no runtime effect — just evaluate the constructor.
                self.eval_expr(ctor, env)
            }

            ExprKind::KernelLaunch { config, kernel } => {
                self.eval_kernel_launch(config, kernel, env)
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

            ExprKind::TryElseBlock(try_stmts, else_stmts) => self.eval_expr_try_else_block(try_stmts, else_stmts, env),

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
                Ok(Value::Array(vals.into()))
            }

            ExprKind::ArrayFill { value, count } => {
                let cv = self.eval_expr(count, Rc::clone(&env))?;
                let n = match cv { Value::Int(n) => n as usize, Value::Uint(n) => n as usize, _ => return Err(Signal::Error(RuntimeError { message: "array count must be int".into(), line: expr.line, col: 0, len: 0 })) };
                let v = self.eval_expr(value, Rc::clone(&env))?;
                Ok(Value::Array(vec![v; n].into()))
            }

            ExprKind::ArrayAlloc { count } => {
                let cv = self.eval_expr(count, Rc::clone(&env))?;
                let n = match cv { Value::Int(n) => n as usize, Value::Uint(n) => n as usize, _ => return Err(Signal::Error(RuntimeError { message: "array count must be int".into(), line: expr.line, col: 0, len: 0 })) };
                Ok(Value::Array(vec![Value::Int(0); n].into()))
            }

            ExprKind::ArrayComp { expr, var, count } => {
                let cv = self.eval_expr(count, Rc::clone(&env))?;
                let n = match cv { Value::Int(n) => n as usize, Value::Uint(n) => n as usize, _ => return Err(Signal::Error(RuntimeError { message: "array count must be int".into(), line: expr.line, col: 0, len: 0 })) };
                let mut vals = Vec::with_capacity(n);
                for i in 0..n {
                    let inner = Env::child(Rc::clone(&env));
                    inner.borrow_mut().define(var, Value::Int(i as i64));
                    vals.push(self.eval_expr(expr, Rc::clone(&inner))?);
                }
                Ok(Value::Array(vals.into()))
            }

            ExprKind::ArrayCompIter { expr, var, iter } => {
                let col = self.eval_expr(iter, Rc::clone(&env))?;
                let elems: Vec<Value> = match col {
                    Value::Array(v) => Rc::try_unwrap(v).unwrap_or_else(|rc| (*rc).clone()),
                    _ => return Err(Signal::Error(RuntimeError { message: "array comprehension source must be an array".into(), line: expr.line, col: 0, len: 0 })),
                };
                let mut vals = Vec::with_capacity(elems.len());
                for item in elems {
                    let inner = Env::child(Rc::clone(&env));
                    inner.borrow_mut().define(var, item);
                    vals.push(self.eval_expr(expr, Rc::clone(&inner))?);
                }
                Ok(Value::Array(vals.into()))
            }

            // Labeled multi-dim array nodes are lowered away before evaluation ever
            // sees them: `desugar_labeled_array` rewrites LabeledIndex/RelabelCast to
            // plain Index/passthrough and LabeledArrayComp to ArrayAlloc + nested for
            // loops for CPU-side code; `lower_labeled_array_methods` (eval_gpu.rs)
            // does the fixed-shape kernel-field equivalent. Reaching one of these here
            // means a desugar/lowering pass was skipped — an internal compiler bug,
            // not a user-facing error. See docs/array-multidim-proposal.md.
            ExprKind::LabeledIndex(..) | ExprKind::LabeledArrayComp { .. } | ExprKind::RelabelCast(..) => {
                Err(err("internal error: labeled multi-dim array expression reached the evaluator without being desugared first", expr.line))
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
                let mut val = self.eval_expr(inner, Rc::clone(&env))?;
                // Auto-invoke `def ()` when `task obj` and obj is a callable struct with no params.
                if let Value::Object(_) = &val {
                    let mut out_self = None;
                    val = self.call_method(val, "", vec![], line, &mut out_self)?;
                }
                // Invalidate owned captures — the task is now the sole owner
                for name in &owned_captures {
                    env.borrow_mut().invalidate(name);
                }
                Ok(Value::Future(Box::new(val)))
            }

            ExprKind::TaskWithTimeout(_dur, inner) => {
                // In the interpreter there is no real async runtime, so the timeout
                // is silently ignored — same as the existing `timeout()` stub.
                // The body is evaluated eagerly and wrapped in a Future.
                let owned_captures = Self::collect_owned_task_captures(inner, &env);
                Self::check_task_captures(inner, &env, line)?;
                let val = self.eval_expr(inner, Rc::clone(&env))?;
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
                    let v = self.eval_expr(e, Rc::clone(&env))?;
                    // Unwrap Future wrappers produced by `task expr`
                    let v = match v {
                        Value::Future(inner) => *inner,
                        other => other,
                    };
                    results.push(v);
                }
                Ok(Value::Tuple(results))
            }

            ExprKind::GenericCall(callee, _type_args, args) => self.eval_expr_generic_call(callee, args, env, line),

            ExprKind::SliceRange { .. } => {
                Err(err("SliceRange cannot appear outside an index expression", line))
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
    pub(crate) fn eval_binop(&mut self, op: &BinOp, lhs: &Expr, rhs: &Expr, env: EnvRef, line: usize, col: usize) -> Eval {
        let (lcol, llen) = (lhs.col, lhs.len);
        let (rcol, rlen) = (rhs.col, rhs.len);
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
            // Each arithmetic/bitwise op is dispatched to its own method rather than inlined
            // here — with 12 numeric kinds, a single function holding all ten ops' match arms
            // inline gave `eval_binop` a huge debug-mode stack frame (every arm's temporaries
            // count toward the frame regardless of which branch runs), which blew the stack on
            // deeply-recursive struct-method call chains. Splitting scopes each op's frame to
            // only when that op actually executes.
            BinOp::Add => self.eval_add(l, r, line, lcol, llen, rcol, rlen),
            BinOp::Sub => self.eval_sub(l, r, line, lcol, llen, rcol, rlen),
            BinOp::Mul => self.eval_mul(l, r, line, lcol, llen, rcol, rlen),
            BinOp::Div => self.eval_div(l, r, line, lcol, llen, rcol, rlen),
            BinOp::Rem => self.eval_rem(l, r, line, lcol, llen, rcol, rlen),
            BinOp::BitAnd => self.eval_bitand(l, r, line, lcol, llen, rcol, rlen),
            BinOp::BitOr => self.eval_bitor(l, r, line, lcol, llen, rcol, rlen),
            BinOp::BitXor => self.eval_bitxor(l, r, line, lcol, llen, rcol, rlen),
            BinOp::Shl => self.eval_shl(l, r, line, lcol, llen, rcol, rlen),
            BinOp::Shr => self.eval_shr(l, r, line, lcol, llen, rcol, rlen),
            BinOp::Eq => {
                if let Some(result) = self.try_operator_method(&l, "eq", r.clone(), line)? {
                    Ok(result)
                } else {
                    Ok(Value::Bool(Self::values_equal(&l, &r)))
                }
            }
            BinOp::RefEq => {
                // Reference equality — bypasses user-defined eq, always compares identity
                let result = match (&l, &r) {
                    (Value::Object(a), Value::Object(b)) => Rc::ptr_eq(a, b),
                    _ => Self::values_equal(&l, &r),   // primitives have value semantics, identity == equality
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
                    Ok(Value::Bool(!Self::values_equal(&l, &r)))
                }
            }
            BinOp::Lt => {
                if let Some(result) = self.try_operator_method(&l, "lt", r.clone(), line)? {
                    Ok(result)
                } else {
                    self.compare_values(l, r, |o| o == std::cmp::Ordering::Less, line, col)
                }
            }
            BinOp::Gt => {
                if let Some(result) = self.try_operator_method(&l, "gt", r.clone(), line)? {
                    Ok(result)
                } else {
                    self.compare_values(l, r, |o| o == std::cmp::Ordering::Greater, line, col)
                }
            }
            BinOp::LtEq => {
                if let Some(result) = self.try_operator_method(&l, "le", r.clone(), line)? {
                    Ok(result)
                } else {
                    self.compare_values(l, r, |o| o != std::cmp::Ordering::Greater, line, col)
                }
            }
            BinOp::GtEq => {
                if let Some(result) = self.try_operator_method(&l, "ge", r.clone(), line)? {
                    Ok(result)
                } else {
                    self.compare_values(l, r, |o| o != std::cmp::Ordering::Less, line, col)
                }
            }
            BinOp::And | BinOp::Or => unreachable!(),
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

    // Each of the following ten methods holds one arithmetic/bitwise operator's full
    // match arms (see the doc comment on the `eval_binop` dispatch above for why these
    // are split into separate functions rather than inlined into one big match).

    #[allow(clippy::too_many_arguments)]
    fn eval_add(&mut self, l: Value, r: Value, line: usize, lcol: usize, llen: usize, rcol: usize, rlen: usize) -> Eval {
        match (l, r) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_add(b))),
            (Value::Uint(a), Value::Uint(b)) => Ok(Value::Uint(a.wrapping_add(b))),
            (Value::Uint8(a), Value::Uint8(b)) => Ok(Value::Uint8(a.wrapping_add(b))),
            (Value::Uint(a), Value::Int(b)) => {
                if b < 0 { Err(err_span("cannot add negative Int to Uint", line, rcol, rlen)) }
                else { Ok(Value::Uint(a.wrapping_add(b as u64))) }
            }
            (Value::Int(a), Value::Uint(b)) => {
                if a < 0 { Err(err_span("cannot add negative Int to Uint", line, rcol, rlen)) }
                else { Ok(Value::Uint((a as u64).wrapping_add(b))) }
            }
            (Value::Uint8(a), Value::Uint(b)) => Ok(Value::Uint((a as u64).wrapping_add(b))),
            (Value::Uint(a), Value::Uint8(b)) => Ok(Value::Uint(a.wrapping_add(b as u64))),
            (Value::Int8(a), Value::Int8(b)) => Ok(Value::Int8(a.wrapping_add(b))),
            (Value::Int16(a), Value::Int16(b)) => Ok(Value::Int16(a.wrapping_add(b))),
            (Value::Int32(a), Value::Int32(b)) => Ok(Value::Int32(a.wrapping_add(b))),
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a.wrapping_add(b))),
            (Value::Int128(a), Value::Int128(b)) => Ok(Value::Int128(a.wrapping_add(b))),
            (Value::Uint16(a), Value::Uint16(b)) => Ok(Value::Uint16(a.wrapping_add(b))),
            (Value::Uint32(a), Value::Uint32(b)) => Ok(Value::Uint32(a.wrapping_add(b))),
            (Value::Uint64(a), Value::Uint64(b)) => Ok(Value::Uint64(a.wrapping_add(b))),
            (Value::Uint128(a), Value::Uint128(b)) => Ok(Value::Uint128(a.wrapping_add(b))),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 + b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + b as f64)),
            (Value::Str(a), Value::Str(b)) => Ok(Value::Str(a + &b)),
            (Value::Array(a), Value::Array(b)) => {
                let mut new_vec = Rc::try_unwrap(a).unwrap_or_else(|rc| (*rc).clone());
                new_vec.extend(b.iter().cloned());
                Ok(Value::Array(new_vec.into()))
            }
            (a, b) => {
                if let Some(result) = eval_numeric_mixed(&a, &b, &BinOp::Add, line, rcol, rlen) {
                    return result;
                }
                // Try user-defined operator overload first (e.g. `Vec2 + Vec2`)
                if let Some(result) = self.try_operator_method(&a, "add", b.clone(), line)? {
                    return Ok(result);
                }
                // RustType / opaque Object arithmetic: Instant + Duration, etc.
                // These are stubs in the interpreter (no real time); return the left
                // operand so `let deadline = Instant.now() + dur` yields an opaque value
                // that timeout/wait stubs can ignore.
                if matches!(&a, Value::RustType { .. } | Value::Object { .. }) {
                    return Ok(a);
                }
                Err(err_span(format!("cannot add {} and {}", a.type_name(), b.type_name()), line, lcol, llen))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_sub(&mut self, l: Value, r: Value, line: usize, lcol: usize, llen: usize, rcol: usize, rlen: usize) -> Eval {
        match (l, r) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_sub(b))),
            (Value::Uint(a), Value::Uint(b)) => {
                if b > a { Err(err_span("uint subtraction underflow", line, rcol, rlen)) }
                else { Ok(Value::Uint(a - b)) }
            }
            (Value::Uint8(a), Value::Uint8(b)) => {
                if b > a { Err(err_span("uint8 subtraction underflow", line, rcol, rlen)) }
                else { Ok(Value::Uint8(a - b)) }
            }
            (Value::Uint(a), Value::Int(b)) => {
                if b < 0 { Ok(Value::Uint(a.wrapping_add((-b) as u64))) }
                else if (b as u64) > a { Err(err_span("uint subtraction underflow", line, rcol, rlen)) }
                else { Ok(Value::Uint(a - b as u64)) }
            }
            (Value::Int(a), Value::Uint(b)) => Ok(Value::Int(a - b as i64)),
            (Value::Uint8(a), Value::Uint(b)) => {
                let au = a as u64;
                if b > au { Err(err_span("uint subtraction underflow", line, rcol, rlen)) }
                else { Ok(Value::Uint(au - b)) }
            }
            (Value::Uint(a), Value::Uint8(b)) => {
                let bu = b as u64;
                if bu > a { Err(err_span("uint subtraction underflow", line, rcol, rlen)) }
                else { Ok(Value::Uint(a - bu)) }
            }
            (Value::Int8(a), Value::Int8(b)) => Ok(Value::Int8(a.wrapping_sub(b))),
            (Value::Int16(a), Value::Int16(b)) => Ok(Value::Int16(a.wrapping_sub(b))),
            (Value::Int32(a), Value::Int32(b)) => Ok(Value::Int32(a.wrapping_sub(b))),
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a.wrapping_sub(b))),
            (Value::Int128(a), Value::Int128(b)) => Ok(Value::Int128(a.wrapping_sub(b))),
            (Value::Uint16(a), Value::Uint16(b)) => {
                if b > a { Err(err_span("uint16 subtraction underflow", line, rcol, rlen)) } else { Ok(Value::Uint16(a - b)) }
            }
            (Value::Uint32(a), Value::Uint32(b)) => {
                if b > a { Err(err_span("uint32 subtraction underflow", line, rcol, rlen)) } else { Ok(Value::Uint32(a - b)) }
            }
            (Value::Uint64(a), Value::Uint64(b)) => {
                if b > a { Err(err_span("uint64 subtraction underflow", line, rcol, rlen)) } else { Ok(Value::Uint64(a - b)) }
            }
            (Value::Uint128(a), Value::Uint128(b)) => {
                if b > a { Err(err_span("uint128 subtraction underflow", line, rcol, rlen)) } else { Ok(Value::Uint128(a - b)) }
            }
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 - b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - b as f64)),
            (a, b) => {
                if let Some(result) = eval_numeric_mixed(&a, &b, &BinOp::Sub, line, rcol, rlen) {
                    return result;
                }
                if let Some(result) = self.try_operator_method(&a, "sub", b.clone(), line)? {
                    Ok(result)
                } else {
                    Err(err_span(format!("cannot subtract {} and {}", a.type_name(), b.type_name()), line, lcol, llen))
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_mul(&mut self, l: Value, r: Value, line: usize, lcol: usize, llen: usize, rcol: usize, rlen: usize) -> Eval {
        match (l, r) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.wrapping_mul(b))),
            (Value::Uint(a), Value::Uint(b)) => Ok(Value::Uint(a.wrapping_mul(b))),
            (Value::Uint8(a), Value::Uint8(b)) => Ok(Value::Uint8(a.wrapping_mul(b))),
            (Value::Uint(a), Value::Int(b)) => {
                if b < 0 { Err(err_span("cannot multiply Uint by negative Int", line, rcol, rlen)) }
                else { Ok(Value::Uint(a.wrapping_mul(b as u64))) }
            }
            (Value::Int(a), Value::Uint(b)) => {
                if a < 0 { Err(err_span("cannot multiply Uint by negative Int", line, rcol, rlen)) }
                else { Ok(Value::Uint((a as u64).wrapping_mul(b))) }
            }
            (Value::Uint8(a), Value::Uint(b)) => Ok(Value::Uint((a as u64).wrapping_mul(b))),
            (Value::Uint(a), Value::Uint8(b)) => Ok(Value::Uint(a.wrapping_mul(b as u64))),
            (Value::Int8(a), Value::Int8(b)) => Ok(Value::Int8(a.wrapping_mul(b))),
            (Value::Int16(a), Value::Int16(b)) => Ok(Value::Int16(a.wrapping_mul(b))),
            (Value::Int32(a), Value::Int32(b)) => Ok(Value::Int32(a.wrapping_mul(b))),
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a.wrapping_mul(b))),
            (Value::Int128(a), Value::Int128(b)) => Ok(Value::Int128(a.wrapping_mul(b))),
            (Value::Uint16(a), Value::Uint16(b)) => Ok(Value::Uint16(a.wrapping_mul(b))),
            (Value::Uint32(a), Value::Uint32(b)) => Ok(Value::Uint32(a.wrapping_mul(b))),
            (Value::Uint64(a), Value::Uint64(b)) => Ok(Value::Uint64(a.wrapping_mul(b))),
            (Value::Uint128(a), Value::Uint128(b)) => Ok(Value::Uint128(a.wrapping_mul(b))),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 * b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * b as f64)),
            (a, b) => {
                if let Some(result) = eval_numeric_mixed(&a, &b, &BinOp::Mul, line, rcol, rlen) {
                    return result;
                }
                if let Some(result) = self.try_operator_method(&a, "mul", b.clone(), line)? {
                    Ok(result)
                } else {
                    Err(err_span(format!("cannot multiply {} and {}", a.type_name(), b.type_name()), line, lcol, llen))
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_div(&mut self, l: Value, r: Value, line: usize, lcol: usize, llen: usize, rcol: usize, rlen: usize) -> Eval {
        match (l, r) {
            (Value::Int(a), Value::Int(b)) => {
                if b == 0 { Err(err_span("division by zero", line, rcol, rlen)) } else { Ok(Value::Int(a / b)) }
            }
            (Value::Uint(a), Value::Uint(b)) => {
                Ok(Value::Uint(a.checked_div(b).ok_or_else(|| err_span("division by zero", line, rcol, rlen))?))
            }
            (Value::Uint8(a), Value::Uint8(b)) => {
                Ok(Value::Uint8(a.checked_div(b).ok_or_else(|| err_span("division by zero", line, rcol, rlen))?))
            }
            (Value::Uint(a), Value::Int(b)) => {
                if b == 0 { Err(err_span("division by zero", line, rcol, rlen)) }
                else if b < 0 { Err(err_span("cannot divide Uint by negative Int", line, rcol, rlen)) }
                else { Ok(Value::Uint(a / b as u64)) }
            }
            (Value::Int(a), Value::Uint(b)) => {
                if b == 0 { Err(err_span("division by zero", line, rcol, rlen)) }
                else { Ok(Value::Int(a / b as i64)) }
            }
            (Value::Uint8(a), Value::Uint(b)) => {
                Ok(Value::Uint((a as u64).checked_div(b).ok_or_else(|| err_span("division by zero", line, rcol, rlen))?))
            }
            (Value::Uint(a), Value::Uint8(b)) => {
                Ok(Value::Uint(a.checked_div(b as u64).ok_or_else(|| err_span("division by zero", line, rcol, rlen))?))
            }
            (Value::Int8(a), Value::Int8(b)) => Ok(Value::Int8(a.checked_div(b).ok_or_else(|| err_span("division by zero", line, rcol, rlen))?)),
            (Value::Int16(a), Value::Int16(b)) => Ok(Value::Int16(a.checked_div(b).ok_or_else(|| err_span("division by zero", line, rcol, rlen))?)),
            (Value::Int32(a), Value::Int32(b)) => Ok(Value::Int32(a.checked_div(b).ok_or_else(|| err_span("division by zero", line, rcol, rlen))?)),
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a.checked_div(b).ok_or_else(|| err_span("division by zero", line, rcol, rlen))?)),
            (Value::Int128(a), Value::Int128(b)) => Ok(Value::Int128(a.checked_div(b).ok_or_else(|| err_span("division by zero", line, rcol, rlen))?)),
            (Value::Uint16(a), Value::Uint16(b)) => Ok(Value::Uint16(a.checked_div(b).ok_or_else(|| err_span("division by zero", line, rcol, rlen))?)),
            (Value::Uint32(a), Value::Uint32(b)) => Ok(Value::Uint32(a.checked_div(b).ok_or_else(|| err_span("division by zero", line, rcol, rlen))?)),
            (Value::Uint64(a), Value::Uint64(b)) => Ok(Value::Uint64(a.checked_div(b).ok_or_else(|| err_span("division by zero", line, rcol, rlen))?)),
            (Value::Uint128(a), Value::Uint128(b)) => Ok(Value::Uint128(a.checked_div(b).ok_or_else(|| err_span("division by zero", line, rcol, rlen))?)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(a as f64 / b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / b as f64)),
            (a, b) => {
                if let Some(result) = eval_numeric_mixed(&a, &b, &BinOp::Div, line, rcol, rlen) {
                    return result;
                }
                if let Some(result) = self.try_operator_method(&a, "div", b.clone(), line)? {
                    Ok(result)
                } else {
                    Err(err_span(format!("cannot divide {} and {}", a.type_name(), b.type_name()), line, lcol, llen))
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_rem(&mut self, l: Value, r: Value, line: usize, lcol: usize, llen: usize, rcol: usize, rlen: usize) -> Eval {
        match (l, r) {
            (Value::Int(a), Value::Int(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) } else { Ok(Value::Int(a % b)) }
            }
            (Value::Uint(a), Value::Uint(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) } else { Ok(Value::Uint(a % b)) }
            }
            (Value::Uint8(a), Value::Uint8(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) } else { Ok(Value::Uint8(a % b)) }
            }
            (Value::Uint(a), Value::Int(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) }
                else if b < 0 { Err(err_span("cannot take remainder of Uint by negative Int", line, rcol, rlen)) }
                else { Ok(Value::Uint(a % b as u64)) }
            }
            (Value::Int(a), Value::Uint(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) }
                else { Ok(Value::Int(a % b as i64)) }
            }
            (Value::Uint8(a), Value::Uint(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) } else { Ok(Value::Uint((a as u64) % b)) }
            }
            (Value::Uint(a), Value::Uint8(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) } else { Ok(Value::Uint(a % (b as u64))) }
            }
            (Value::Int8(a), Value::Int8(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) } else { Ok(Value::Int8(a % b)) }
            }
            (Value::Int16(a), Value::Int16(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) } else { Ok(Value::Int16(a % b)) }
            }
            (Value::Int32(a), Value::Int32(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) } else { Ok(Value::Int32(a % b)) }
            }
            (Value::Int64(a), Value::Int64(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) } else { Ok(Value::Int64(a % b)) }
            }
            (Value::Int128(a), Value::Int128(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) } else { Ok(Value::Int128(a % b)) }
            }
            (Value::Uint16(a), Value::Uint16(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) } else { Ok(Value::Uint16(a % b)) }
            }
            (Value::Uint32(a), Value::Uint32(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) } else { Ok(Value::Uint32(a % b)) }
            }
            (Value::Uint64(a), Value::Uint64(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) } else { Ok(Value::Uint64(a % b)) }
            }
            (Value::Uint128(a), Value::Uint128(b)) => {
                if b == 0 { Err(err_span("remainder by zero", line, rcol, rlen)) } else { Ok(Value::Uint128(a % b)) }
            }
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a % b)),
            (Value::Int(a),   Value::Float(b)) => Ok(Value::Float((a as f64) % b)),
            (Value::Float(a), Value::Int(b))   => Ok(Value::Float(a % (b as f64))),
            (a, b) => {
                if let Some(result) = eval_numeric_mixed(&a, &b, &BinOp::Rem, line, rcol, rlen) {
                    return result;
                }
                if let Some(result) = self.try_operator_method(&a, "rem", b.clone(), line)? {
                    Ok(result)
                } else {
                    Err(err_span(format!("cannot remainder {} and {}", a.type_name(), b.type_name()), line, lcol, llen))
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_bitand(&mut self, l: Value, r: Value, line: usize, lcol: usize, llen: usize, rcol: usize, rlen: usize) -> Eval {
        match (l, r) {
            (Value::Int(a),  Value::Int(b))  => Ok(Value::Int(a & b)),
            (Value::Uint(a), Value::Uint(b)) => Ok(Value::Uint(a & b)),
            (Value::Uint8(a), Value::Uint8(b)) => Ok(Value::Uint8(a & b)),
            (Value::Int8(a), Value::Int8(b)) => Ok(Value::Int8(a & b)),
            (Value::Int16(a), Value::Int16(b)) => Ok(Value::Int16(a & b)),
            (Value::Int32(a), Value::Int32(b)) => Ok(Value::Int32(a & b)),
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a & b)),
            (Value::Int128(a), Value::Int128(b)) => Ok(Value::Int128(a & b)),
            (Value::Uint16(a), Value::Uint16(b)) => Ok(Value::Uint16(a & b)),
            (Value::Uint32(a), Value::Uint32(b)) => Ok(Value::Uint32(a & b)),
            (Value::Uint64(a), Value::Uint64(b)) => Ok(Value::Uint64(a & b)),
            (Value::Uint128(a), Value::Uint128(b)) => Ok(Value::Uint128(a & b)),
            (Value::Uint(a), Value::Int(b))  => Ok(Value::Uint(a & b as u64)),
            (Value::Int(a),  Value::Uint(b)) => Ok(Value::Int(a & b as i64)),
            (a, b) => {
                if let Some(result) = eval_numeric_mixed(&a, &b, &BinOp::BitAnd, line, rcol, rlen) { return result; }
                Err(err_span(format!("cannot bitwise-and {} and {}", a.type_name(), b.type_name()), line, lcol, llen))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_bitor(&mut self, l: Value, r: Value, line: usize, lcol: usize, llen: usize, rcol: usize, rlen: usize) -> Eval {
        match (l, r) {
            (Value::Int(a),  Value::Int(b))  => Ok(Value::Int(a | b)),
            (Value::Uint(a), Value::Uint(b)) => Ok(Value::Uint(a | b)),
            (Value::Uint8(a), Value::Uint8(b)) => Ok(Value::Uint8(a | b)),
            (Value::Int8(a), Value::Int8(b)) => Ok(Value::Int8(a | b)),
            (Value::Int16(a), Value::Int16(b)) => Ok(Value::Int16(a | b)),
            (Value::Int32(a), Value::Int32(b)) => Ok(Value::Int32(a | b)),
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a | b)),
            (Value::Int128(a), Value::Int128(b)) => Ok(Value::Int128(a | b)),
            (Value::Uint16(a), Value::Uint16(b)) => Ok(Value::Uint16(a | b)),
            (Value::Uint32(a), Value::Uint32(b)) => Ok(Value::Uint32(a | b)),
            (Value::Uint64(a), Value::Uint64(b)) => Ok(Value::Uint64(a | b)),
            (Value::Uint128(a), Value::Uint128(b)) => Ok(Value::Uint128(a | b)),
            (Value::Uint(a), Value::Int(b))  => Ok(Value::Uint(a | b as u64)),
            (Value::Int(a),  Value::Uint(b)) => Ok(Value::Int(a | b as i64)),
            (a, b) => {
                if let Some(result) = eval_numeric_mixed(&a, &b, &BinOp::BitOr, line, rcol, rlen) { return result; }
                Err(err_span(format!("cannot bitwise-or {} and {}", a.type_name(), b.type_name()), line, lcol, llen))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_bitxor(&mut self, l: Value, r: Value, line: usize, lcol: usize, llen: usize, rcol: usize, rlen: usize) -> Eval {
        match (l, r) {
            (Value::Int(a),  Value::Int(b))  => Ok(Value::Int(a ^ b)),
            (Value::Uint(a), Value::Uint(b)) => Ok(Value::Uint(a ^ b)),
            (Value::Uint8(a), Value::Uint8(b)) => Ok(Value::Uint8(a ^ b)),
            (Value::Int8(a), Value::Int8(b)) => Ok(Value::Int8(a ^ b)),
            (Value::Int16(a), Value::Int16(b)) => Ok(Value::Int16(a ^ b)),
            (Value::Int32(a), Value::Int32(b)) => Ok(Value::Int32(a ^ b)),
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a ^ b)),
            (Value::Int128(a), Value::Int128(b)) => Ok(Value::Int128(a ^ b)),
            (Value::Uint16(a), Value::Uint16(b)) => Ok(Value::Uint16(a ^ b)),
            (Value::Uint32(a), Value::Uint32(b)) => Ok(Value::Uint32(a ^ b)),
            (Value::Uint64(a), Value::Uint64(b)) => Ok(Value::Uint64(a ^ b)),
            (Value::Uint128(a), Value::Uint128(b)) => Ok(Value::Uint128(a ^ b)),
            (Value::Uint(a), Value::Int(b))  => Ok(Value::Uint(a ^ b as u64)),
            (Value::Int(a),  Value::Uint(b)) => Ok(Value::Int(a ^ b as i64)),
            (a, b) => {
                if let Some(result) = eval_numeric_mixed(&a, &b, &BinOp::BitXor, line, rcol, rlen) { return result; }
                Err(err_span(format!("cannot bitwise-xor {} and {}", a.type_name(), b.type_name()), line, lcol, llen))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_shl(&mut self, l: Value, r: Value, line: usize, lcol: usize, llen: usize, rcol: usize, rlen: usize) -> Eval {
        match (l, r) {
            (Value::Int(a),  Value::Int(b))  if b >= 0 => Ok(Value::Int(a.wrapping_shl(b as u32))),
            (Value::Uint(a), Value::Int(b))  if b >= 0 => Ok(Value::Uint(a.wrapping_shl(b as u32))),
            (Value::Uint(a), Value::Uint(b)) => Ok(Value::Uint(a.wrapping_shl(b as u32))),
            (Value::Uint8(a), Value::Int(b)) if b >= 0 => Ok(Value::Uint8(a.wrapping_shl(b as u32))),
            (Value::Uint8(a), Value::Uint8(b)) => Ok(Value::Uint8(a.wrapping_shl(b as u32))),
            (Value::Int8(a), Value::Int8(b)) => Ok(Value::Int8(a.wrapping_shl(b as u32))),
            (Value::Int16(a), Value::Int16(b)) => Ok(Value::Int16(a.wrapping_shl(b as u32))),
            (Value::Int32(a), Value::Int32(b)) => Ok(Value::Int32(a.wrapping_shl(b as u32))),
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a.wrapping_shl(b as u32))),
            (Value::Int128(a), Value::Int128(b)) => Ok(Value::Int128(a.wrapping_shl(b as u32))),
            (Value::Uint16(a), Value::Uint16(b)) => Ok(Value::Uint16(a.wrapping_shl(b as u32))),
            (Value::Uint32(a), Value::Uint32(b)) => Ok(Value::Uint32(a.wrapping_shl(b))),
            (Value::Uint64(a), Value::Uint64(b)) => Ok(Value::Uint64(a.wrapping_shl(b as u32))),
            (Value::Uint128(a), Value::Uint128(b)) => Ok(Value::Uint128(a.wrapping_shl(b as u32))),
            (_, Value::Int(b)) if b < 0 => Err(err_span("shift amount cannot be negative", line, rcol, rlen)),
            (a, b) => {
                if let Some(result) = eval_numeric_mixed(&a, &b, &BinOp::Shl, line, rcol, rlen) { return result; }
                Err(err_span(format!("cannot shift {} by {}", a.type_name(), b.type_name()), line, lcol, llen))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_shr(&mut self, l: Value, r: Value, line: usize, lcol: usize, llen: usize, rcol: usize, rlen: usize) -> Eval {
        match (l, r) {
            (Value::Int(a),  Value::Int(b))  if b >= 0 => Ok(Value::Int(a.wrapping_shr(b as u32))),
            (Value::Uint(a), Value::Int(b))  if b >= 0 => Ok(Value::Uint(a.wrapping_shr(b as u32))),
            (Value::Uint(a), Value::Uint(b)) => Ok(Value::Uint(a.wrapping_shr(b as u32))),
            (Value::Uint8(a), Value::Int(b)) if b >= 0 => Ok(Value::Uint8(a.wrapping_shr(b as u32))),
            (Value::Uint8(a), Value::Uint8(b)) => Ok(Value::Uint8(a.wrapping_shr(b as u32))),
            (Value::Int8(a), Value::Int8(b)) => Ok(Value::Int8(a.wrapping_shr(b as u32))),
            (Value::Int16(a), Value::Int16(b)) => Ok(Value::Int16(a.wrapping_shr(b as u32))),
            (Value::Int32(a), Value::Int32(b)) => Ok(Value::Int32(a.wrapping_shr(b as u32))),
            (Value::Int64(a), Value::Int64(b)) => Ok(Value::Int64(a.wrapping_shr(b as u32))),
            (Value::Int128(a), Value::Int128(b)) => Ok(Value::Int128(a.wrapping_shr(b as u32))),
            (Value::Uint16(a), Value::Uint16(b)) => Ok(Value::Uint16(a.wrapping_shr(b as u32))),
            (Value::Uint32(a), Value::Uint32(b)) => Ok(Value::Uint32(a.wrapping_shr(b))),
            (Value::Uint64(a), Value::Uint64(b)) => Ok(Value::Uint64(a.wrapping_shr(b as u32))),
            (Value::Uint128(a), Value::Uint128(b)) => Ok(Value::Uint128(a.wrapping_shr(b as u32))),
            (_, Value::Int(b)) if b < 0 => Err(err_span("shift amount cannot be negative", line, rcol, rlen)),
            (a, b) => {
                if let Some(result) = eval_numeric_mixed(&a, &b, &BinOp::Shr, line, rcol, rlen) { return result; }
                Err(err_span(format!("cannot shift {} by {}", a.type_name(), b.type_name()), line, lcol, llen))
            }
        }
    }

    // Cross-numeric-type equality (Int/Uint/Uint8/Float), matching the promotions
    // `compare_values` applies for `<`/`>`/etc. — otherwise derived PartialEq treats
    // e.g. Value::Int(5) and Value::Uint(5) as unequal since they're different variants.
    pub(crate) fn values_equal(l: &Value, r: &Value) -> bool {
        // Float mixed with any integer kind — compare as f64 (generalizes the old
        // Int/Uint/Uint8-only arms to all 12 numeric kinds, fixing the pre-existing gap
        // where Uint8 vs Float wasn't wired in here).
        if let Value::Float(a) = l {
            if let Some(b) = value_as_i128(r) { return *a == b as f64; }
        }
        if let Value::Float(b) = r {
            if let Some(a) = value_as_i128(l) { return a as f64 == *b; }
        }
        // Two different integer kinds — compare via i128 staging (same-kind pairs fall
        // through to the derived `l == r` below, which handles full-width Uint128/Int128).
        if let (Some(ka), Some(kb)) = (NumKind::of(l), NumKind::of(r)) {
            if ka != kb {
                if let (Some(a), Some(b)) = (value_as_i128(l), value_as_i128(r)) {
                    return a == b;
                }
            }
        }
        l == r
    }

    pub(crate) fn compare_values(&self, l: Value, r: Value, pred: impl Fn(std::cmp::Ordering) -> bool, line: usize, col: usize) -> Eval {
        let ord = if let (Value::Float(a), Value::Float(b)) = (&l, &r) {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        } else if let Value::Float(a) = &l {
            match value_as_i128(&r) {
                Some(b) => a.partial_cmp(&(b as f64)).unwrap_or(std::cmp::Ordering::Equal),
                None => return Err(err_at(format!("cannot compare {} and {}", l.type_name(), r.type_name()), line, col)),
            }
        } else if let Value::Float(b) = &r {
            match value_as_i128(&l) {
                Some(a) => (a as f64).partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                None => return Err(err_at(format!("cannot compare {} and {}", l.type_name(), r.type_name()), line, col)),
            }
        } else if let (Value::Str(a), Value::Str(b)) = (&l, &r) {
            a.cmp(b)
        } else if let (Value::Uint128(a), Value::Uint128(b)) = (&l, &r) {
            a.cmp(b)
        } else if let (Value::Int128(a), Value::Int128(b)) = (&l, &r) {
            a.cmp(b)
        } else if NumKind::of(&l).is_some() && NumKind::of(&r).is_some() {
            match (value_as_i128(&l), value_as_i128(&r)) {
                (Some(a), Some(b)) => a.cmp(&b),
                _ => return Err(err_at(format!("cannot compare {} and {}", l.type_name(), r.type_name()), line, col)),
            }
        } else {
            return Err(err_at(format!("cannot compare {} and {}", l.type_name(), r.type_name()), line, col));
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
                func(&args, line)
            }
            Value::Fn { decl, captured } => {
                self.call_fn(&decl, captured, args, line, in_throws_context)
            }
            Value::OverloadedFn { name, variants } => {
                // Find the first variant whose parameter types match the given args.
                let chosen = variants.iter().find(|(decl, _)| {
                    if decl.params.len() != args.len() { return false; }
                    decl.params.iter().zip(args.iter()).all(|(param, arg)| {
                        match &param.ty {
                            None => true,
                            Some(ty) => {
                                let resolved = self.resolve_type(ty);
                                self.value_matches_type_simple(arg, &resolved)
                            }
                        }
                    })
                });
                match chosen {
                    Some((decl, captured)) => {
                        let decl = decl.clone();
                        let captured = Rc::clone(captured);
                        self.call_fn(&decl, captured, args, line, in_throws_context)
                    }
                    None => Err(err(format!("no matching overload for '{}'", name), line)),
                }
            }
            Value::Closure { params, body, captured } => {
                self.call_closure(params, body, captured, args, line)
            }
            Value::Struct { decl, captured } => {
                // Struct is callable as constructor
                let obj = self.instantiate_struct_labeled(&decl, &captured, args, line)?;
                Ok(obj)
            }
            Value::KernelStruct { decl, captured } => {
                // Kernel struct callable as constructor: calls its `init` if present.
                self.instantiate_kernel_struct(&decl, &captured, args, line)
            }
            Value::EnumNamespace { name, variants: _, .. } => {
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
            Value::Object(_) => {
                // Callable struct: look for anonymous `def ()` / `req ()` method (name == "")
                let mut out_self = None;
                let result = self.call_method(callee.clone(), "", args, line, &mut out_self)?;
                Ok(result)
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
            "Vec" | "VecDeque" => Ok(Value::Array(args.into())),
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

    pub(crate) fn item_pub_name(item: &Item) -> Option<(&str, bool)> {
        match item {
            Item::Fn(d)     => Some((&d.name, d.is_pub)),
            Item::Struct(d) => Some((&d.name, d.is_pub)),
            Item::Enum(d)   => Some((&d.name, d.is_pub)),
            Item::Let(d)    => Some((&d.name, d.is_pub)),
            Item::Mod(d)    => Some((&d.name, false)),
            Item::Kernel(d) => Some((&d.name, d.is_pub)),
            Item::Use(_) | Item::Alias(_) | Item::Trait(_) | Item::Ext(_) | Item::Stmt(_) => None,
        }
    }

    pub(crate) fn exec_use_decl(&mut self, decl: &UseDecl, env: EnvRef) -> Result<(), Signal> {
        let path = &decl.path;
        if path.is_empty() {
            return Ok(());
        }
        // Single-component path (e.g. `use token.*`) — go straight to filesystem loader.
        if path.len() < 2 {
            return self.exec_use(decl, env);
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
