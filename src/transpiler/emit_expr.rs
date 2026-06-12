use super::*;
use super::Transpiler;
use super::helpers::*;

impl Transpiler {
    pub(crate) fn emit_expr(&self, expr: &Expr) -> String {
        match &expr.kind {
            ExprKind::Int(n)   => n.to_string(),
            ExprKind::Float(f) => {
                let s = format!("{}", f);
                if s.contains('.') || s.contains('e') || s.contains('E') { s }
                else { format!("{}.0", s) }
            }
            ExprKind::Str(s) => format!("\"{}\"", escape_str(s)),
            ExprKind::StringInterp(segs) => self.emit_interp(segs),
            ExprKind::Bool(b)  => b.to_string(),
            ExprKind::Nil      => "None".into(),
            ExprKind::Void     => "()".into(),
            ExprKind::Var(n)   => {
                // In init body, `self` is the local `__self` variable.
                if self.in_init_body && n == "self" {
                    return "__self".to_string();
                }
                // Implicit self: inside a struct method, a bare field name maps to `self.field`
                // only when it is NOT already declared as a local variable.
                if let Some(struct_name) = &self.self_type {
                    if !self.known_local_vars.contains(n.as_str()) {
                        if let Some(fields) = self.struct_fields.get(struct_name.as_str()) {
                            if fields.iter().any(|(f, _)| f == n) {
                                let self_ref = if self.in_init_body { "__self" } else { "self" };
                                return format!("{}.{}", self_ref, escape_rust_keyword(n));
                            }
                        }
                    }
                }
                self.map_builtin_var(n)
            }

            ExprKind::BinOp(op, l, r) => {
                // Reference equality ===
                if matches!(op, BinOp::RefEq) {
                    let ls = self.emit_expr(l);
                    let rs = self.emit_expr(r);
                    return format!("Arc::ptr_eq(&{}, &{})", ls, rs);
                }
                // `x is SomeType` / `x is not SomeType` — type/nil check
                if matches!(op, BinOp::Is | BinOp::IsNot) {
                    let is_not = matches!(op, BinOp::IsNot);
                    // `x is nil` / `x is not nil` — right side is Nil
                    if matches!(r.kind, ExprKind::Nil) {
                        // Check if left side is an optional variable
                        let is_optional = matches!(&l.kind, ExprKind::Var(v) if
                            self.optional_vars.contains(v.as_str()));
                        if is_optional {
                            let ls = self.emit_expr(l);
                            return if is_not {
                                format!("({} != None)", ls)
                            } else {
                                format!("({} == None)", ls)
                            };
                        }
                        // Left side is `None` literal — comparing nil to nil
                        if matches!(l.kind, ExprKind::Nil) {
                            return if is_not { "false".to_string() } else { "true".to_string() };
                        }
                        // Non-optional value: `x is nil` is always false, `x is not nil` always true
                        return if is_not { "true".to_string() } else { "false".to_string() };
                    }
                    // `x is y` — reference identity between Rc-wrapped struct variables
                    if let (ExprKind::Var(lv), ExprKind::Var(rv)) = (&l.kind, &r.kind) {
                        if self.rc_identity_vars.contains(lv.as_str())
                            && self.rc_identity_vars.contains(rv.as_str())
                        {
                            return if is_not {
                                format!("(!Rc::ptr_eq(&{}, &{}))", lv, rv)
                            } else {
                                format!("(Rc::ptr_eq(&{}, &{}))", lv, rv)
                            };
                        }
                    }
                    // `x is TypeName` — struct type check
                    if let ExprKind::Var(type_name) = &r.kind {
                        if self.struct_fields.contains_key(type_name.as_str()) {
                            let ls = self.emit_expr(l);
                            return if is_not {
                                format!("!matches!({}, {} {{ .. }})", ls, type_name)
                            } else {
                                format!("matches!({}, {} {{ .. }})", ls, type_name)
                            };
                        }
                        // Enum variant check — `x is EnumVariant` (unit variant)
                        if let Some(enum_name) = self.enum_variants.get(type_name.as_str()) {
                            let ls = self.emit_expr(l);
                            let qualified = format!("{}::{}", enum_name, type_name);
                            return if is_not {
                                format!("!matches!({}, {})", ls, qualified)
                            } else {
                                format!("matches!({}, {})", ls, qualified)
                            };
                        }
                    }
                }
                // String concatenation: if either side is a string expression, emit as Arc::<str>::from(format!(...))
                // This handles: string literal, string interp, known string vars, and nested string +.
                if matches!(op, BinOp::Add) && (self.is_string_expr(l) || self.is_string_expr(r)) {
                    // Flatten the whole chain into a single format! call.
                    let mut parts: Vec<String> = Vec::new();
                    self.collect_string_parts(expr, &mut parts);
                    let fmt = parts.iter().map(|_| "{}").collect::<Vec<_>>().join("");
                    return format!("Arc::<str>::from(format!(\"{}\", {}))", fmt, parts.join(", "));
                }
                // Numeric type coercion: when adding/subtracting/multiplying typed numeric vars
                // of different widths (i8 + i16, etc.), cast both to the wider type.
                // Also handle mixed float-literal/int-literal arithmetic: `7.5 % 2` → `7.5_f64 % 2_f64`.
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem) {
                    let l_is_float_lit = matches!(l.kind, ExprKind::Float(_));
                    let r_is_float_lit = matches!(r.kind, ExprKind::Float(_));
                    let l_is_int_lit   = matches!(l.kind, ExprKind::Int(_));
                    let r_is_int_lit   = matches!(r.kind, ExprKind::Int(_));
                    if (l_is_float_lit && r_is_int_lit) || (l_is_int_lit && r_is_float_lit) {
                        let ls_raw = self.emit_expr(l);
                        let rs_raw = self.emit_expr(r);
                        let ls = if l_is_int_lit { format!("{}_f64", ls_raw) } else { ls_raw };
                        let rs = if r_is_int_lit { format!("{}_f64", rs_raw) } else { rs_raw };
                        return format!("({} {} {})", ls, binop_str(op), rs);
                    }
                    if let Some((l_ty, r_ty)) = self.get_numeric_types(l, r) {
                        if l_ty != r_ty {
                            let wider = wider_numeric_type(&l_ty, &r_ty);
                            let ls_raw = self.emit_expr(l);
                            let rs_raw = self.emit_expr(r);
                            let ls = if l_ty != wider { format!("({} as {})", ls_raw, wider) } else { ls_raw };
                            let rs = if r_ty != wider { format!("({} as {})", rs_raw, wider) } else { rs_raw };
                            return format!("({} {} {})", ls, binop_str(op), rs);
                        }
                    }
                }
                // Struct operator method dispatch: `a + b` → `a.clone().add(b.clone())`
                // when the left operand's struct type has an operator method registered.
                let method_name = match op {
                    BinOp::Add   => Some("add"),
                    BinOp::Sub   => Some("sub"),
                    BinOp::Mul   => Some("mul"),
                    BinOp::Div   => Some("div"),
                    BinOp::Rem   => Some("rem"),
                    BinOp::Eq    => Some("eq"),
                    BinOp::NotEq => Some("ne"),
                    BinOp::Lt    => Some("lt"),
                    BinOp::LtEq  => Some("le"),
                    BinOp::Gt    => Some("gt"),
                    BinOp::GtEq  => Some("ge"),
                    _ => None,
                };
                if let Some(mname) = method_name {
                    // Determine struct type from left operand.
                    let struct_ty = if let ExprKind::Var(vname) = &l.kind {
                        self.var_struct_types.get(vname.as_str()).cloned()
                    } else {
                        None
                    };
                    if let Some(sty) = struct_ty {
                        let key = format!("{}::{}", sty, mname);
                        if self.struct_operator_methods.contains(&key) {
                            let ls = self.emit_expr(l);
                            // Look up param types to decide if rhs needs Box::new() wrapping.
                            let param_types = self.struct_operator_param_types.get(&key).cloned();
                            let rs_raw = self.emit_expr(r);
                            let rs = if let Some(ptypes) = param_types {
                                if let Some(pty) = ptypes.first() {
                                    if matches!(pty, Type::Qualified(_, OwnerQual::Owned)) {
                                        // Need to clone before boxing to avoid moving `rs_raw`
                                        // when it's used multiple times (e.g. e3 == e3).
                                        let clone_expr = if rs_raw.ends_with(".clone()") {
                                            rs_raw.clone()
                                        } else {
                                            format!("{}.clone()", rs_raw)
                                        };
                                        // In managed mode: wrap in Arc<Mutex<T>> or RefCell<T>.
                                        if self.is_managed_owned_user(pty) {
                                            match self.config.threading {
                                                crate::transpiler::ThreadingMode::Multi =>
                                                    format!("Arc::new(std::sync::Mutex::new({}))", clone_expr),
                                                crate::transpiler::ThreadingMode::Single =>
                                                    format!("RefCell::new({})", clone_expr),
                                            }
                                        } else {
                                            format!("Box::new({})", clone_expr)
                                        }
                                    } else {
                                        rs_raw
                                    }
                                } else {
                                    rs_raw
                                }
                            } else {
                                rs_raw
                            };
                            return format!("{}.clone().{}({})", ls, mname, rs);
                        }
                    }
                }
                // Arc<str> equality: wrap any string literal in Arc::<str>::from(...) when
                // compared with a non-literal expression (which may be Arc<str>).
                // This ensures type compatibility without needing full type inference.
                if matches!(op, BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq) {
                    let l_is_raw_lit = matches!(&l.kind, ExprKind::Str(_));
                    let r_is_raw_lit = matches!(&r.kind, ExprKind::Str(_));
                    // is_arc_str: non-literal side is known Arc<str>; literals alone don't count.
                    let l_is_arc_str = !l_is_raw_lit && self.is_string_expr(l);
                    let r_is_arc_str = !r_is_raw_lit && self.is_string_expr(r);
                    // Check if either side is a &str (str-ref) typed variable — skip wrapping.
                    let l_is_str_ref = matches!(&l.kind, ExprKind::Var(v)
                        if self.var_types.get(v.as_str()).map(Self::is_str_ref_type).unwrap_or(false));
                    let r_is_str_ref = matches!(&r.kind, ExprKind::Var(v)
                        if self.var_types.get(v.as_str()).map(Self::is_str_ref_type).unwrap_or(false));
                    // Wrap into Arc<str> when any side is a string literal or known Arc<str>,
                    // unless a side is a &str variable (which rejects Arc<str> comparisons).
                    let should_wrap = !l_is_str_ref && !r_is_str_ref
                        && (l_is_raw_lit || r_is_raw_lit || l_is_arc_str || r_is_arc_str);
                    if should_wrap {
                        let ls_expr = self.emit_expr(l);
                        let rs_expr = self.emit_expr(r);
                        // For ordering ops: use &str deref so PartialOrd<str> kicks in.
                        let ord_op = match op {
                            BinOp::Lt    => Some("< std::cmp::Ordering::Equal"),
                            BinOp::LtEq  => Some("!= std::cmp::Ordering::Greater"),
                            BinOp::Gt    => Some("> std::cmp::Ordering::Equal"),
                            BinOp::GtEq  => Some("!= std::cmp::Ordering::Less"),
                            _ => None,
                        };
                        if let Some(ord) = ord_op {
                            let l_deref = if l_is_raw_lit {
                                if let ExprKind::Str(s) = &l.kind { format!("\"{}\"", escape_str(s)) }
                                else { ls_expr.clone() }
                            } else { format!("(&*{})", ls_expr) };
                            let r_deref = if r_is_raw_lit {
                                if let ExprKind::Str(s) = &r.kind { format!("\"{}\"", escape_str(s)) }
                                else { rs_expr.clone() }
                            } else { format!("(&*{})", rs_expr) };
                            return format!("({}.cmp({}) {})", l_deref, r_deref, ord);
                        }
                        // For Eq/NotEq: wrap literals in Arc::<str>::from(...) for type compat.
                        let ls = if l_is_raw_lit {
                            if let ExprKind::Str(s) = &l.kind {
                                format!("Arc::<str>::from(\"{}\")", escape_str(s))
                            } else { ls_expr.clone() }
                        } else { ls_expr };
                        let rs = if r_is_raw_lit {
                            if let ExprKind::Str(s) = &r.kind {
                                format!("Arc::<str>::from(\"{}\")", escape_str(s))
                            } else { rs_expr.clone() }
                        } else { rs_expr };
                        return format!("({} {} {})", ls, binop_str(op), rs);
                    }
                }
                let ls = self.emit_expr(l);
                let rs = self.emit_expr(r);
                format!("({} {} {})", ls, binop_str(op), rs)
            }
            ExprKind::UnaryOp(op, e) => {
                let s = self.emit_expr(e);
                // Struct unary neg dispatch: `-a` → `a.clone().neg()`
                if matches!(op, UnaryOp::Neg) {
                    if let ExprKind::Var(vname) = &e.kind {
                        if let Some(sty) = self.var_struct_types.get(vname.as_str()).cloned() {
                            let key = format!("{}::neg", sty);
                            if self.struct_operator_methods.contains(&key) {
                                return format!("{}.clone().neg()", s);
                            }
                        }
                    }
                }
                match op {
                    UnaryOp::Neg    => format!("(-{})", s),
                    UnaryOp::Not    => format!("(!{})", s),
                    UnaryOp::BitNot => format!("(!{})", s),
                }
            }
            ExprKind::Assign(target, value) => {
                // Global mutable var assignment: `logX = val` → `*LOGX.lock().unwrap() = val`.
                if let ExprKind::Var(var_name) = &target.kind {
                    if self.global_vars_used_in_fns.contains(var_name.as_str()) {
                        let static_name = var_name.to_uppercase();
                        let val_s = self.emit_expr_owned(value);
                        return format!("*{}.lock().unwrap_or_else(|e| e.into_inner()) = {}", static_name, val_s);
                    }
                }
                if let ExprKind::Field(obj, field) = &target.kind {
                    // Instance setter property: `t.prop = v` → `t.set_prop(v)`.
                    // Check if `field` is registered as a setter for any struct.
                    let is_instance_setter = self.struct_setters.iter()
                        .any(|k| k.ends_with(&format!("::{}", field)));
                    if is_instance_setter {
                        let obj_s = self.emit_expr(obj);
                        let val_s = self.emit_expr_owned(value);
                        return format!("{}.set_{}({})", obj_s, field, val_s);
                    }
                    // If assigning to a type var that has a type setter, call the setter function.
                    if let ExprKind::Var(type_name) = &obj.kind {
                        if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                            let key = format!("{}::{}", type_name, field);
                            if self.struct_type_mut_var_names.contains(&key) {
                                // Invoke the type setter if one exists, unless already inside it
                                let has_setter = !self.in_type_setter
                                    && self.struct_type_method_sigs.get(type_name.as_str())
                                        .and_then(|m| m.get(field.as_str()))
                                        .map(|k| matches!(k, TypeMethodKind::Set))
                                        .unwrap_or(false);
                                if has_setter {
                                    let val_s = self.emit_expr_owned(value);
                                    return format!("{}::set_{}({})", type_name, field, val_s);
                                }
                            }
                        }
                    }
                }
                // Transient field write: self.field = v → self.field.set(v) (Cell) or *self.field.borrow_mut() = v (RefCell)
                // When the field type is Optional, the assigned value is coerced to Some(v).
                if let ExprKind::Field(obj, field) = &target.kind {
                    if let ExprKind::Var(v) = &obj.kind {
                        if v == "self" {
                            let key = self.self_type.as_deref()
                                .map(|t| format!("{}::{}", t, field));
                            if let Some(k) = key {
                                if let Some((is_copy, field_ty, _)) = self.transient_fields.get(&k) {
                                    let is_copy = *is_copy;
                                    // Wrap in Some() if field is Optional and value is not nil
                                    let raw_val = self.emit_expr_owned(value);
                                    let is_nil = matches!(value.kind, ExprKind::Nil);
                                    let val_s = if !is_nil && matches!(field_ty, Type::Optional(_)) {
                                        if raw_val.starts_with("Some(") || raw_val == "None" {
                                            raw_val
                                        } else {
                                            format!("Some({})", raw_val)
                                        }
                                    } else {
                                        raw_val
                                    };
                                    return if is_copy {
                                        format!("self.{}.set({})", field, val_s)
                                    } else {
                                        format!("*self.{}.borrow_mut() = {}", field, val_s)
                                    };
                                }
                            }
                        }
                    }
                }
                // Mutex field write: w.field = v
                if let ExprKind::Field(obj, field) = &target.kind {
                    if let ExprKind::Var(v) = &obj.kind {
                        if self.var_mutex_types.contains(v.as_str()) {
                            let val_s = self.emit_expr_owned(value);
                            let guard = self.actor_write_guard(v);
                            return format!("{{ let mut __g = {}; __g.{} = {}; }}", guard, field, val_s);
                        }
                    }
                    // self.worker.field = v
                    if let ExprKind::Field(inner_obj, mutex_field) = &obj.kind {
                        if let ExprKind::Var(v) = &inner_obj.kind {
                            if v == "self" {
                                let key = self.self_type.as_deref()
                                    .map(|t| format!("{}::{}", t, mutex_field));
                                if let Some(k) = key {
                                    if self.struct_mutex_fields.contains(&k) {
                                        let val_s = self.emit_expr_owned(value);
                                        let guard = self.actor_write_guard(&format!("self.{}", mutex_field));
                                        return format!("{{ let mut __g = {}; __g.{} = {}; }}", guard, field, val_s);
                                    }
                                }
                            }
                        }
                    }
                }
                // RwLock field write: c.field = v
                if let ExprKind::Field(obj, field) = &target.kind {
                    if let ExprKind::Var(v) = &obj.kind {
                        if self.var_rwlock_types.contains(v.as_str()) {
                            let val_s = self.emit_expr_owned(value);
                            let guard = self.guard_write_guard(v);
                            return format!("{{ let mut __wg = {}; __wg.{} = {}; }}", guard, field, val_s);
                        }
                    }
                    // self.data.field = v
                    if let ExprKind::Field(inner_obj, rwlock_field) = &obj.kind {
                        if let ExprKind::Var(v) = &inner_obj.kind {
                            if v == "self" {
                                let key = self.self_type.as_deref()
                                    .map(|t| format!("{}::{}", t, rwlock_field));
                                if let Some(k) = key {
                                    if self.struct_rwlock_fields.contains(&k) {
                                        let val_s = self.emit_expr_owned(value);
                                        let guard = self.guard_write_guard(&format!("self.{}", rwlock_field));
                                        return format!("{{ let mut __wg = {}; __wg.{} = {}; }}", guard, field, val_s);
                                    }
                                }
                            }
                        }
                    }
                }
                // Compound assignment: `x = x op rhs` → `x op= rhs` (idiomatic Rust).
                // Detected by matching BinOp(op, lhs_copy, rhs) where lhs_copy emits the same
                // string as target — safe because the parser already desugared `x op= rhs`.
                // Exception: string addition (`Arc<str>` does not implement `AddAssign`).
                if let ExprKind::BinOp(op, lhs_copy, rhs) = &value.kind {
                    let is_string_add = matches!(op, BinOp::Add)
                        && (matches!(lhs_copy.kind, ExprKind::Str(_) | ExprKind::StringInterp(_))
                            || matches!(rhs.kind, ExprKind::Str(_) | ExprKind::StringInterp(_))
                            || matches!(&target.kind, ExprKind::Var(v)
                                if self.arc_vars.contains(v.as_str()) || self.string_arc_vars.contains(v.as_str())));
                    if !is_string_add {
                        let compound_op = match op {
                            BinOp::Add => Some("+="),
                            BinOp::Sub => Some("-="),
                            BinOp::Mul => Some("*="),
                            BinOp::Div => Some("/="),
                            BinOp::Rem => Some("%="),
                            _ => None,
                        };
                        if let Some(op_str) = compound_op {
                            let target_s = self.emit_expr(target);
                            let lhs_s    = self.emit_expr(lhs_copy);
                            if target_s == lhs_s {
                                let rhs_s = self.emit_expr_owned(rhs);
                                return format!("{} {} {}", target_s, op_str, rhs_s);
                            }
                        }
                    }
                }
                // Dict subscript assignment: dict[key] = val → dict.insert(key_owned, val)
                if let ExprKind::Index(dict_obj, key) = &target.kind {
                    if let ExprKind::Var(dict_name) = &dict_obj.kind {
                        if self.dict_vars.contains(dict_name.as_str()) {
                            let key_owned = self.emit_dict_key_owned(key);
                            let val_s = self.emit_expr_owned(value);
                            return format!("{}.insert({}, {})", dict_name, key_owned, val_s);
                        }
                    }
                }
                // emit_expr_owned wraps string literals in Arc::from; falls through for other types
                // For index LHS (arr[i] = v), emit without .clone() since we're writing not reading.
                let target_s = if let ExprKind::Index(arr_obj, idx_expr) = &target.kind {
                    if let ExprKind::Var(arr_name) = &arr_obj.kind {
                        if !self.dict_vars.contains(arr_name.as_str()) {
                            let raw_idx = self.emit_expr(idx_expr);
                            let idx_s = match &idx_expr.kind {
                                ExprKind::Int(_) | ExprKind::Var(_) | ExprKind::BinOp(..) | ExprKind::Field(..) => format!("({}) as usize", raw_idx),
                                _ => raw_idx,
                            };
                            format!("{}[{}]", arr_name, idx_s)
                        } else {
                            self.emit_expr(target)
                        }
                    } else {
                        self.emit_expr(target)
                    }
                } else {
                    self.emit_expr(target)
                };
                format!("{} = {}", target_s, self.emit_expr_owned(value))
            }
            ExprKind::Field(obj, field) => {
                // Special case: `(task expr).value` / `(task expr).wait` where the task body
                // captures non-Arc local variables.  We cannot safely `tokio::spawn(async move {})`
                // because that would move the variable — leaving the outer scope without it.
                // Solution: inline the async call instead of spawning.
                if field == "value" || field == "wait" {
                    if let ExprKind::Task(inner_e) = &obj.kind {
                        let captured = collect_var_names(inner_e);
                        let has_non_arc_captures = captured.iter().any(|v| {
                            self.known_local_vars.contains(v.as_str())
                                && !self.arc_vars.contains(v.as_str())
                                && !self.string_arc_vars.contains(v.as_str())
                                && !self.weak_vars.contains(v.as_str())
                        });
                        if has_non_arc_captures {
                            // Inline: emit the inner expression (method call already gets .await
                            // appended by emit_expr for async methods).
                            let inner_s = self.emit_expr(inner_e);
                            return if field == "wait" {
                                format!("{{ let _ = {}; }}", inner_s)
                            } else {
                                inner_s
                            };
                        }
                    }
                }
                // Type-level access: `Counter.MAX` → `Counter::MAX`, `Counter.count` → `Counter::count()`
                if let ExprKind::Var(type_name) = &obj.kind {
                    if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        let key = format!("{}::{}", type_name, field);
                        if self.struct_type_var_names.contains(&key) {
                            // type let → associated const (UPPER_CASE in Rust)
                            return format!("{}::{}", type_name, field.to_uppercase());
                        }
                        if self.struct_type_mut_var_names.contains(&key) {
                            // type var → module-level static Mutex: read via lock(), recover from poisoning
                            return format!("*{}.lock().unwrap_or_else(|e| e.into_inner())", field.to_uppercase());
                        }
                        // Fieldless enum variant (no args): CalcError.DivByZero → CalcError::DivByZero
                        if self.enum_variant_fields.contains_key(&key) {
                            return format!("{}::{}", type_name, field);
                        }
                        // Fallback for external PascalCase types (Ordering, Duration, etc.):
                        // `Ordering.SeqCst` → `Ordering::SeqCst`
                        if !self.known_local_vars.contains(type_name.as_str()) {
                            return format!("{}::{}", type_name, field);
                        }
                    }
                    // oneshot rx.value → rx.await.unwrap() (receive the single value)
                    if field == "value" && self.oneshot_receivers.contains(type_name.as_str()) {
                        return if self.in_throws || self.in_try_body {
                            format!("{}.await?", type_name)
                        } else {
                            format!("{}.await.unwrap()", type_name)
                        };
                    }
                    // watch rx.value → current value without waiting
                    if field == "value" && self.watch_receivers.contains(type_name.as_str()) {
                        return format!("{}.borrow().clone()", type_name);
                    }
                    // future.value / future.wait on a spawned JoinHandle or async param.
                    //
                    // Three cases:
                    //  (a) throws JoinHandle  — JoinHandle<Result<T,BoringError>>
                    //      throws ctx : f.await.unwrap()?      — unwrap JoinError, propagate inner
                    //      plain ctx  : f.await.unwrap().unwrap()  — panic on inner error
                    //  (b) plain JoinHandle   — JoinHandle<T>
                    //      always     : f.await.unwrap()       — just unwrap JoinError
                    //  (c) async fn param     — impl Future<Output=T> (or Result<T,_>)
                    //      throws ctx + value : f.await?
                    //      otherwise          : f.await.unwrap()
                    if field == "done" && self.task_vars.contains(type_name.as_str()) {
                        return format!(
                            "tokio::time::timeout(std::time::Duration::ZERO, {}).await.is_ok()",
                            type_name
                        );
                    }
                    if (field == "value" || field == "wait") && self.task_vars.contains(type_name.as_str()) {
                        let in_throws_ctx = self.in_throws || self.in_try_body;
                        let is_throws_handle = self.throws_join_handle_vars.contains(type_name.as_str());
                        let is_join_handle   = self.join_handle_vars.contains(type_name.as_str());
                        return if field == "wait" {
                            if is_throws_handle && in_throws_ctx {
                                format!("{{ let _ = {}.await.map_err(|__e| Box::new(BoringError::String(Arc::from(__e.to_string()))) as Box<dyn std::error::Error + Send + Sync>)??.expect(\"unhandled task error\"); }}", type_name)
                            } else if is_throws_handle {
                                format!("{{ let _ = {}.await.expect(\"task panicked\").expect(\"unhandled task error\"); }}", type_name)
                            } else if in_throws_ctx {
                                format!("{{ {}.await.map_err(|__e| Box::new(BoringError::String(Arc::from(__e.to_string()))) as Box<dyn std::error::Error + Send + Sync>)?; }}", type_name)
                            } else {
                                format!("{{ let _ = {}.await; }}", type_name)
                            }
                        } else {
                            // .value
                            if is_throws_handle {
                                if in_throws_ctx {
                                    format!("{}.await.expect(\"task panicked\")?", type_name)
                                } else {
                                    format!("{}.await.expect(\"task panicked\").expect(\"unhandled task error\")", type_name)
                                }
                            } else if is_join_handle {
                                format!("{}.await.expect(\"task panicked\")", type_name)
                            } else if in_throws_ctx {
                                format!("{}.await?", type_name)
                            } else {
                                format!("{}.await.expect(\"task panicked\")", type_name)
                            }
                        };
                    }
                }
                let obj_s = self.emit_expr(obj);
                // `.value` / `.wait` on a JoinHandle → `.await.unwrap()`.
                // Covers inline task expressions `(task ...).value` and loop vars `future.wait`
                // that aren't tracked in task_vars.
                // Future.done() — non-blocking poll: true if the JoinHandle is finished.
                if field == "done" {
                    // Use try_join with zero timeout as a non-blocking poll.
                    return format!(
                        "tokio::time::timeout(std::time::Duration::ZERO, {}).await.is_ok()",
                        obj_s
                    );
                }
                if field == "value" || field == "wait" {
                    // TaskWithTimeout: always a throws JoinHandle (wraps Result<T, Elapsed>).
                    // Needs .await.unwrap()? in throws context to propagate Error.Expired,
                    // or .await.unwrap().unwrap() otherwise (panics on Elapsed).
                    if matches!(&obj.kind, ExprKind::TaskWithTimeout(..)) {
                        let in_throws_ctx = self.in_throws || self.in_try_body;
                        return if field == "wait" {
                            if in_throws_ctx {
                                format!("{{ let _ = {}.await.expect(\"task panicked\")?; }}", obj_s)
                            } else {
                                format!("{{ let _ = {}.await.expect(\"task panicked\").expect(\"unhandled task error\"); }}", obj_s)
                            }
                        } else if in_throws_ctx {
                            format!("{}.await.expect(\"task panicked\")?", obj_s)
                        } else {
                            format!("{}.await.expect(\"task panicked\").expect(\"unhandled task error\")", obj_s)
                        };
                    }

                    let is_future = matches!(&obj.kind, ExprKind::Task(_))
                        || obj_s.contains("tokio::spawn")
                        || obj_s.contains("async move");
                    if is_future {
                        return if field == "wait" {
                            format!("{{ let _ = {}.await; }}", obj_s)
                        } else {
                            format!("{}.await.expect(\"task panicked\")", obj_s)
                        };
                    }
                    // Loop variable holding a JoinHandle: only treat as future if the var is
                    // explicitly in task_vars (declared with a task expression).
                    // Using var_struct_types, var_types, or struct_fields to distinguish
                    // struct field access from JoinHandle avoids false positives on plain
                    // structs with a "value" field (e.g. LetStmt, ReturnStmt, pair tuples).
                    if let ExprKind::Var(v) = &obj.kind {
                        let is_known_struct = self.var_struct_types.contains_key(v.as_str())
                            || self.struct_fields.contains_key(v.as_str())
                            || self.var_types.get(v.as_str()).map(|t| {
                                if let Type::Named(tn) = t { self.struct_fields.contains_key(tn.as_str()) } else { false }
                            }).unwrap_or(false);
                        let is_task = self.task_vars.contains(v.as_str());
                        if is_task
                            && v != "self"
                            && !self.var_mutex_types.contains(v.as_str())
                            && !is_known_struct
                        {
                            return if field == "wait" {
                                format!("{{ let _ = {}.await; }}", obj_s)
                            } else {
                                format!("{}.await.expect(\"task panicked\")", obj_s)
                            };
                        }
                    }
                }
                // Check if this field access is a getter property (req method with no params).
                // (a) `self.field` where `self` is the current struct instance and `field` is a getter.
                // (b) `var.field` where `var` is any variable, and `field` is registered as a getter
                //     in any struct — cross-struct fallback for `let t = Temperature(); t.fahrenheit`.
                // Both guards require obj to be a plain Var (not a chained field access like `self.text`)
                // to avoid incorrectly treating built-in properties (`.length` on strings/arrays).
                let is_getter = if let ExprKind::Var(v) = &obj.kind {
                    if v == "self" {
                        self.self_type.as_deref()
                            .map(|t| self.struct_getters.contains(&format!("{}::{}", t, field)))
                            .unwrap_or(false)
                    } else {
                        let from_struct = self.var_struct_types.get(v.as_str())
                            .map(|type_name| self.struct_getters.contains(&format!("{}::{}", type_name, field)))
                            .unwrap_or(false);
                        let from_enum = if !from_struct {
                            if let Some(Type::Named(type_name)) = self.var_types.get(v.as_str()) {
                                self.struct_getters.contains(&format!("{}::{}", type_name, field))
                            } else { false }
                        } else { false };
                        from_struct || from_enum
                    }
                } else {
                    false
                };
                // Enum field accessors return Option<T> — unwrap at callsite with a clear message.
                let is_enum_field_getter = if let ExprKind::Var(v) = &obj.kind {
                    let type_name = if v == "self" {
                        self.self_type.clone()
                    } else {
                        self.var_types.get(v.as_str()).and_then(|t| {
                            if let Type::Named(n) = t { Some(n.clone()) } else { None }
                        })
                    };
                    type_name.map(|t| self.enum_field_getters.contains(&format!("{}::{}", t, field)))
                        .unwrap_or(false)
                } else {
                    false
                };
                if is_enum_field_getter {
                    return format!("{}.{}().expect(\"field '{}' not available in this variant\")", obj_s, field, field);
                }
                if is_getter {
                    return format!("{}.{}()", obj_s, field);
                }
                // Mutex var access: w.field → w.lock().await.field (multi) or w.borrow().field (single)
                if let ExprKind::Var(v) = &obj.kind {
                    if self.var_mutex_types.contains(v.as_str()) {
                        let access = self.actor_read_access(v);
                        // In single-thread mode, if the field is itself an Rc<RefCell<T>> (actor/guard),
                        // we must Rc::clone it to avoid moving out of the borrow guard.
                        let field_is_rc = matches!(self.config.threading, crate::transpiler::ThreadingMode::Single)
                            && self.var_types.get(v.as_str())
                                .and_then(|t| match t { Type::Named(n) => Some(n.as_str()), Type::Qualified(inner, _) => if let Type::Named(n) = inner.as_ref() { Some(n.as_str()) } else { None }, _ => None })
                                .or_else(|| self.var_struct_types.get(v.as_str()).map(|s| s.as_str()))
                                .and_then(|tname| self.struct_fields.get(tname))
                                .and_then(|fields| fields.iter().find(|(fname, _)| fname == field))
                                .map(|(_, fty)| Self::is_arc_qualified(fty) || Self::is_rc_qualified(fty))
                                .unwrap_or(false);
                        return if field_is_rc {
                            format!("Rc::clone(&{}.{})", access, field)
                        } else {
                            format!("{}.{}", access, field)
                        };
                    }
                    // Managed-mode mutex var (std::sync::Mutex, synchronous):
                    // w.field → w.lock().unwrap().field
                    // If the param has a shadow guard, use it directly (no re-locking).
                    if let Some(shadow) = self.managed_param_shadows.get(v.as_str()) {
                        return format!("{}.{}", shadow, field);
                    }
                    if self.managed_mutex_vars.contains(v.as_str()) {
                        return format!("{}.lock().unwrap().{}", v, field);
                    }
                    // Managed-mode RefCell var (single-thread):
                    // w.field → w.borrow().field
                    if self.managed_refcell_vars.contains(v.as_str()) {
                        return format!("{}.borrow().{}", v, field);
                    }
                }
                // Mutex struct field: self.worker.field → self.worker.lock().await.field (multi) / self.worker.borrow().field (single)
                if let ExprKind::Field(inner_obj, mutex_field) = &obj.kind {
                    if let ExprKind::Var(v) = &inner_obj.kind {
                        if v == "self" {
                            let key = self.self_type.as_deref()
                                .map(|t| format!("{}::{}", t, mutex_field));
                            if let Some(k) = key {
                                if self.struct_mutex_fields.contains(&k) {
                                    return format!("{}.{}", self.actor_read_access(&format!("self.{}", mutex_field)), field);
                                }
                            }
                        }
                    }
                }
                // Transient field read: self.field → self.field.get() (Cell) or self.field.borrow().clone() (RefCell)
                if obj_s == "self" {
                    let key = self.self_type.as_deref()
                        .map(|t| format!("{}::{}", t, field));
                    if let Some(k) = key {
                        if let Some((is_copy, _, _)) = self.transient_fields.get(&k) {
                            return if *is_copy {
                                format!("self.{}.get()", field)
                            } else {
                                format!("self.{}.borrow().clone()", field)
                            };
                        }
                    }
                }
                // Determine if the receiver is a module/type path (use `::`) or instance (use `.`).
                // A receiver is a path when:
                //   (a) it is an uppercase Var (type name like `Ordering`, `Duration`, `File`)
                //   (b) it is a lowercase Var NOT in known_local_vars and NOT `self`
                //       (e.g. `mpsc`, `tokio` — module names imported but not declared as locals)
                //   (c) the emitted receiver already contains `::` (cascaded path: `tokio::time`)
                let is_path_receiver = match &obj.kind {
                    ExprKind::Var(v) => {
                        if v == "self" { false }
                        else if v.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) { true }
                        else { !self.known_local_vars.contains(v.as_str()) }
                    }
                    // A field access on another field/call is a path only if the receiver is also
                    // a plain path (contains `::` but is NOT a call result ending with `)`)
                    _ => obj_s.contains("::") && !obj_s.ends_with(')'),
                };
                if is_path_receiver {
                    return format!("{}::{}", obj_s, field);
                }
                // Don't apply map_field to user-defined struct fields (e.g. a field named
                // `count` should not be remapped to `len()` on a user struct).
                let mapped = if let ExprKind::Var(v) = &obj.kind {
                    let is_user_field = (v == "self")
                        .then(|| self.self_type.as_deref())
                        .flatten()
                        .and_then(|t| self.struct_fields.get(t))
                        .map(|fields| fields.iter().any(|(fname, _)| fname == field))
                        .unwrap_or(false);
                    if is_user_field { field.as_str() } else { map_field(field) }
                } else {
                    map_field(field)
                };
                let result = format!("{}.{}", obj_s, mapped);
                // Wrap `x.len() as i64` etc. in parens to avoid Rust parsing `i64 <` as generic args.
                if mapped.contains(" as ") { format!("({})", result) } else { result }
            }
            ExprKind::Index(obj, idx) => {
                // When the index is an opaque collection index var (Option<usize> from
                // firstIndex/nextIndex), use the get_at(Option<usize>) trait method.
                if let ExprKind::Var(v) = &idx.kind {
                    if self.index_vars.contains(v.as_str()) {
                        return format!("{}.get_at({})", self.emit_expr(obj), v);
                    }
                }
                // Dict vars (HashMap): use .get().cloned().unwrap() for bare access.
                // When wrapped in `else` (ExprKind::Else), that handler rebuilds the get
                // directly as .unwrap_or_else() to avoid a double-unwrap.
                if let ExprKind::Var(obj_var) = &obj.kind {
                    if self.dict_vars.contains(obj_var.as_str()) {
                        let key_ref = self.emit_dict_key_borrow(idx);
                        return format!("{}.get({}).cloned().expect(\"dict key not found\")", obj_var, key_ref);
                    }
                }
                // self.field[key] where field is a dict-type struct field (HashMap): use dict-style access.
                // Detect by checking if the index key is a string-typed var or the field type is Dict.
                if let ExprKind::Field(inner_obj, field_name) = &obj.kind {
                    if let ExprKind::Var(v) = &inner_obj.kind {
                        if v == "self" {
                            let is_dict_field = self.self_type.as_deref()
                                .and_then(|t| self.struct_fields.get(t))
                                .and_then(|fields| fields.iter().find(|(fname, _)| fname == field_name))
                                .map(|(_, fty)| matches!(fty, crate::ast::Type::Dict(..)))
                                .unwrap_or(false);
                            let idx_is_string = match &idx.kind {
                                ExprKind::Var(v) => {
                                    let vt = self.var_types.get(v.as_str());
                                    matches!(vt, Some(crate::ast::Type::Str))
                                    || matches!(vt, Some(crate::ast::Type::Named(n)) if n == "string" || n == "str")
                                    || self.string_vars.contains(v.as_str())
                                }
                                ExprKind::Str(_) => true,
                                _ => false,
                            };
                            if is_dict_field || idx_is_string {
                                let obj_s2 = self.emit_expr(obj);
                                let key_ref = self.emit_dict_key_borrow(idx);
                                return format!("{}.get({}).cloned().expect(\"dict key not found\")", obj_s2, key_ref);
                            }
                        }
                    }
                }
                // For string literal keys (HashMap), use the string key directly (Arc<str>: Deref<Target=str>)
                let idx_s = if matches!(&idx.kind, ExprKind::Str(_)) {
                    format!("&{}", self.emit_expr_owned(idx))
                } else {
                    // Rust requires usize for slice indexing; cast integer expressions.
                    let raw = self.emit_expr(idx);
                    match &idx.kind {
                        ExprKind::Int(_) | ExprKind::Var(_) | ExprKind::BinOp(..) | ExprKind::Field(..) => format!("({}) as usize", raw),
                        _ => raw,
                    }
                };
                // Add .clone() so generic T values can be moved out of collections
                format!("{}[{}].clone()", self.emit_expr(obj), idx_s)
            }
            ExprKind::Call(callee, args) => self.emit_call(callee, args),
            ExprKind::MethodCall(obj, method, args) => self.emit_method_call(obj, method, args),
            ExprKind::Pipe(lhs, name, args) => self.emit_pipe(lhs, name, args),
            ExprKind::GenericCall(callee, type_args, args) => self.emit_generic_call(callee, type_args, args),

            ExprKind::TryElse(e, default) => {
                // `try expr else default` — calls a throws/Result function and returns the Ok
                // value or the default on error.
                // The inner expression must NOT get `?` propagation — TryElse handles the error
                // locally. Use a sub-transpiler with throws flags cleared.
                let mut sub = self.make_sub();
                sub.in_throws = false;
                sub.in_try_body = false;
                let inner = sub.emit_expr(e);
                // `try? expr` desugars to TryElse(expr, Nil) — emit the idiomatic `.ok()`
                // (Result<T,E> → Option<T>) rather than .unwrap_or_else(|_| None).
                if matches!(default.kind, ExprKind::Nil) {
                    return format!("{}.ok()", inner);
                }
                let default_s = self.emit_expr_owned(default);
                format!("{}.unwrap_or_else(|_| {})", inner, default_s)
            }

            ExprKind::TryElseBlock(try_stmts, else_stmts) => {
                // `try … else …` — try/else expression in all four body combinations.
                //
                // Sync context (not inside an async fn):
                //   { match (|| -> Result<_, Box<dyn Error + Send + Sync>> { … })() {
                //       Ok(__boring_v)  => __boring_v,
                //       Err(__boring_e) => { let error = …; <else body> } } }
                //
                // Async context (inside a task/async fn):
                //   { let __boring_r: Result<_, Box<dyn Error + Send + Sync>> =
                //       async { … }.await;
                //     match __boring_r {
                //       Ok(__boring_v)  => __boring_v,
                //       Err(__boring_e) => { let error = …; <else body> } } }
                //
                // The async form avoids the E0728 "await inside non-async closure" error
                // that arises when the try body contains task function calls (.await).
                let mut try_sub = self.make_sub();
                try_sub.in_throws = true;
                try_sub.fn_returns_void = false;
                try_sub.emit_body(try_stmts);

                let mut else_sub = self.make_sub();
                else_sub.in_throws = false;
                else_sub.fn_returns_void = false;
                else_sub.known_local_vars.insert("error".to_string());
                else_sub.emit_body(else_stmts);

                // `error` is bound as the original `Box<dyn Error>`, not as a string.
                // • `{error}` in string interpolation works — Box<dyn Error> implements Display.
                // • `match error:` with string patterns works (compare via Display string).
                // • For typed enum dispatch use `try … catch MyEnum:` which emits the
                //   appropriate downcast_ref automatically.
                if self.in_async {
                    format!(
                        "{{\nlet __boring_r: Result<_, Box<dyn std::error::Error + Send + Sync>> = async {{\n{}}}.await;\nmatch __boring_r {{\nOk(__boring_v) => __boring_v,\nErr(__boring_e) => {{\nlet error = __boring_e;\n{}}},\n}}\n}}",
                        try_sub.out,
                        else_sub.out,
                    )
                } else {
                    format!(
                        "{{\nmatch (|| -> Result<_, Box<dyn std::error::Error + Send + Sync>> {{\n{}}})() {{\nOk(__boring_v) => __boring_v,\nErr(__boring_e) => {{\nlet error = __boring_e;\n{}}},\n}}\n}}",
                        try_sub.out,
                        else_sub.out,
                    )
                }
            }

            ExprKind::Else(e, default) => {
                // `x as T else default` — cast with fallback: always use unwrap_or (never ?)
                if let ExprKind::Cast(inner, ty) = &e.kind {
                    let src = self.emit_expr(inner);
                    let dv = self.emit_expr_owned(default);
                    return match ty {
                        Type::Int => format!("{}.trim().parse::<i64>().unwrap_or({})", src, dv),
                        Type::Uint => format!("{}.trim().parse::<u64>().unwrap_or({})", src, dv),
                        Type::Float => format!("{}.trim().parse::<f64>().unwrap_or({})", src, dv),
                        Type::Bool => format!("({} == \"true\")", src),
                        Type::Named(n) if n == "int" =>
                            format!("{}.trim().parse::<i64>().unwrap_or({})", src, dv),
                        Type::Named(n) if n == "uint" =>
                            format!("{}.trim().parse::<u64>().unwrap_or({})", src, dv),
                        Type::Named(n) if n == "float" =>
                            format!("{}.trim().parse::<f64>().unwrap_or({})", src, dv),
                        Type::Named(n) if n == "bool" => format!("({} == \"true\")", src),
                        _ => format!("{}.unwrap_or({})", self.emit_expr(e), dv),
                    };
                }
                // `dict[key] else default` — rebuild .get() directly to avoid double-unwrap
                // (ExprKind::Index for dict vars emits .unwrap() for bare access).
                if let ExprKind::Index(dict_obj, key) = &e.kind {
                    if let ExprKind::Var(dict_name) = &dict_obj.kind {
                        if self.dict_vars.contains(dict_name.as_str()) {
                            let key_ref = self.emit_dict_key_borrow(key);
                            let dv = self.emit_expr_owned(default);
                            return format!("{}.get({}).cloned().unwrap_or_else(|| {})",
                                dict_name, key_ref, dv);
                        }
                    }
                }
                // `vec[i] else default` — Vec::get returns Option<&T>, use .cloned().unwrap_or_else.
                // Direct indexing would yield T (not Option<T>), so .unwrap_or_else would fail.
                if let ExprKind::Index(arr_obj, idx_expr) = &e.kind {
                    let is_dict = matches!(&arr_obj.kind, ExprKind::Var(v)
                        if self.dict_vars.contains(v.as_str()));
                    if !is_dict {
                        let arr_s = self.emit_expr(arr_obj);
                        let idx_s = self.emit_expr(idx_expr);
                        let dv = self.emit_expr_owned(default);
                        return format!("{}.get(({}) as usize).cloned().unwrap_or_else(|| {})",
                            arr_s, idx_s, dv);
                    }
                }
                // `expr else default` — nil coalescing / Option unwrap
                let e_s = self.emit_expr(e);
                let dv = self.emit_expr_owned(default);
                // When a numeric optional (Option<i64/f64>) is coalesced with a string default,
                // unwrap_or_else won't compile — use map_or_else to convert the value to string.
                let is_numeric_opt_var = matches!(&e.kind, ExprKind::Var(v)
                    if self.optional_numeric_vars.contains(v.as_str()));
                // When an always-None optional is coalesced with a string default, emit default directly.
                let is_always_none = matches!(&e.kind, ExprKind::Var(v)
                    if self.always_none_vars.contains(v.as_str()));
                let rc_ty = if self.use_rc_str() { "Rc" } else { "Arc" };
                let dv_is_str = dv.starts_with("Arc::new(") || dv.starts_with("Arc::<str>::from(")
                    || dv.starts_with("Rc::new(") || dv.starts_with("Rc::<str>::from(");
                if is_always_none && dv_is_str {
                    // This optional is always None — the result is always the default value.
                    dv
                } else if is_numeric_opt_var && dv_is_str {
                    format!("{}.as_ref().map_or_else(|| {}, |v| {rc_ty}::<str>::from(format!(\"{{}}\", v)))", e_s, dv)
                } else {
                    format!("{}.unwrap_or_else(|| {})", e_s, dv)
                }
            }

            ExprKind::Array(elems) => {
                // [TypeName] where TypeName is a known struct/enum → typed empty Vec::<T>::new()
                // This handles Boring's `[T]{}` typed-empty-array idiom (parser splits it into
                // `[T]` + `{}`, and the `{}` becomes a harmless HashSet::new() statement).
                if elems.len() == 1 {
                    if let ExprKind::Var(name) = &elems[0].kind {
                        if self.struct_fields.contains_key(name.as_str())
                            || self.enum_variant_fields.keys().any(|k| {
                                k.starts_with(name.as_str()) && k[name.len()..].starts_with("::")
                            })
                        {
                            return format!("Vec::<{}>::new()", name);
                        }
                    }
                }
                // If any element is a string literal, use emit_expr_owned for all
                // so the vec is typed Vec<Rc<str>> consistently.
                let has_str_lit = elems.iter().any(|e| matches!(&e.kind, ExprKind::Str(_) | ExprKind::StringInterp(_)));
                let es: Vec<String> = elems.iter().map(|e| {
                    if has_str_lit { self.emit_expr_owned(e) } else { self.emit_expr(e) }
                }).collect();
                format!("vec![{}]", es.join(", "))
            }
            ExprKind::Tuple(elems) => {
                // Use emit_expr_owned so string literals become Rc/Arc<str> in tuple slots.
                let es: Vec<String> = elems.iter().map(|e| self.emit_expr_owned(e)).collect();
                format!("({})", es.join(", "))
            }
            ExprKind::Dict(pairs) => {
                if pairs.is_empty() {
                    "HashMap::new()".into()
                } else {
                    // Use emit_expr_owned for both keys and values so string literals
                    // become Arc<str> (string dicts are HashMap<Arc<str>, Arc<str>>).
                    let ps: Vec<String> = pairs.iter()
                        .map(|(k, v)| format!("({}, {})", self.emit_expr_owned(k), self.emit_expr_owned(v)))
                        .collect();
                    format!("HashMap::from([{}])", ps.join(", "))
                }
            }
            ExprKind::Set(elems) => {
                if elems.is_empty() {
                    // Provide a default element type so Rust can infer the HashSet type.
                    "HashSet::<i64>::new()".into()
                } else {
                    let es: Vec<String> = elems.iter().map(|e| self.emit_expr(e)).collect();
                    format!("HashSet::from([{}])", es.join(", "))
                }
            }

            ExprKind::DotIdent(name) => {
                // Enum variant shorthand: `.North` → `Direction::North`
                if let Some(enum_name) = self.enum_variants.get(name) {
                    format!("{}::{}", enum_name, name)
                } else {
                    name.clone() // unknown variant — emit bare name, will be caught by rustc
                }
            }
            ExprKind::Range { start, end, inclusive } => {
                let s = self.emit_expr(start);
                let e = self.emit_expr(end);
                if *inclusive { format!("({}..={})", s, e) } else { format!("({}..{})", s, e) }
            }
            ExprKind::Cast(e, ty) => {
                let src = self.emit_expr(e);
                let dst = self.emit_type(ty);
                // User-defined `as Type:` conversion → call the generated `into_type()` method.
                // Use the lowercased emitted type name for primitive types (float → f64, etc.)
                // as well as named types. Only apply if the source is a struct/enum variable —
                // do not transform string/numeric literal casts.
                let src_is_struct_or_enum = match &e.kind {
                    ExprKind::Var(v) =>
                        self.var_struct_types.contains_key(v.as_str())
                        || self.var_struct_type.contains_key(v.as_str())
                        || matches!(self.var_types.get(v.as_str()),
                            Some(Type::Named(n)) if self.struct_fields.contains_key(n.as_str())),
                    ExprKind::Field(_, _) => true, // field access on struct
                    _ => false,
                };
                let key = match ty {
                    Type::Named(n) => Some(n.to_lowercase()),
                    _ if src_is_struct_or_enum => Some(dst.to_lowercase()),
                    _ => None,
                };
                // Never route `as string` through user_conv_targets — the Display impl's
                // Arc::<str>::from(x.to_string()) path handles it correctly without generating
                // a method name like `into_arc<string>` which is invalid Rust.
                let is_string_cast = matches!(ty, Type::Str)
                    || matches!(ty, Type::Named(n) if n == "string" || n == "str");
                if !is_string_cast {
                    if let Some(k) = key {
                        // Try both the boring type name (e.g. "float") and the Rust type name (e.g. "f64").
                        // user_conv_targets stores the lowercased Rust emit form, but the key from
                        // Type::Named("float") is "float". Try the boring name first, then the emitted form.
                        // Only apply user conversions when the source is a struct/enum instance —
                        // don't call into_f64() on numeric expressions, only on struct variables/fields.
                        if self.user_conv_targets.contains(k.as_str()) {
                            let method = format!("into_{}", k);
                            return format!("{}.{}()", src, method);
                        } else if src_is_struct_or_enum {
                            let rust_key = dst.to_lowercase();
                            if k != rust_key && self.user_conv_targets.contains(rust_key.as_str()) {
                                let method = format!("into_{}", rust_key);
                                return format!("{}.{}()", src, method);
                            }
                        }
                    }
                }
                // Newtype unwrap: `id as uint` where `id` is a known newtype variable → `id.0`.
                // Works for let bindings and function parameters tracked in var_newtype_type.
                if let ExprKind::Var(v) = &e.kind {
                    if let Some(nt_name) = self.var_newtype_type.get(v.as_str()) {
                        if let Some(inner_rust) = self.newtype_inner.get(nt_name.as_str()) {
                            if *inner_rust == dst {
                                return format!("{}.0", src);
                            }
                        }
                    }
                }
                // Newtype construction: `42 as UserId` → `UserId(42)`.
                if let Type::Named(n) = ty {
                    if self.newtype_types.contains(n.as_str()) {
                        return format!("{}({})", n, src);
                    }
                }
                // Cast to Optional type: `s as int?` → parse().ok(), not unwrap_or.
                if let Type::Optional(inner) = ty {
                    let parse_ty = match inner.as_ref() {
                        Type::Int                           => Some("i64"),
                        Type::Uint                          => Some("u64"),
                        Type::Float                         => Some("f64"),
                        Type::Named(n) if n == "int"        => Some("i64"),
                        Type::Named(n) if n == "uint"       => Some("u64"),
                        Type::Named(n) if n == "float"      => Some("f64"),
                        _                                   => None,
                    };
                    return if let Some(pt) = parse_ty {
                        format!("{}.trim().parse::<{}>().ok()", src, pt)
                    } else {
                        format!("{}.try_into().ok()", src)
                    };
                }
                let src_is_numeric_lit = matches!(&e.kind, ExprKind::Int(_) | ExprKind::Float(_));
                let _src_is_bool = matches!(&e.kind, ExprKind::Bool(_))
                    || matches!(&e.kind, ExprKind::Var(v) if {
                        // bool variable (rough heuristic: not in known numeric vars)
                        let _ = v; false
                    });
                let src_is_bool_lit = matches!(&e.kind, ExprKind::Bool(_));
                let src_is_numeric_var = matches!(&e.kind, ExprKind::Var(v)
                    if !self.string_vars.contains(v.as_str()));
                let is_float_ty = matches!(ty, Type::Float)
                    || matches!(ty, Type::Named(n) if n == "float");
                let is_int_ty = matches!(ty, Type::Int | Type::Uint)
                    || matches!(ty, Type::Named(n) if n == "int" || n == "uint");
                let is_bool_ty = matches!(ty, Type::Bool)
                    || matches!(ty, Type::Named(n) if n == "bool");

                // Numeric computation (BinOp/Call/UnaryOp) → numeric cast: use `as T`, not .parse()
                let src_is_expr = matches!(&e.kind,
                    ExprKind::BinOp(_, _, _) | ExprKind::Call(_, _) | ExprKind::UnaryOp(_, _)
                    | ExprKind::MethodCall(_, _, _));
                if src_is_expr && is_float_ty {
                    return format!("({} as f64)", src);
                }
                if src_is_expr && is_int_ty {
                    return format!("({} as i64)", src);
                }
                // Known-numeric variable (tracked in var_types as Int/Float/Uint) → cast with `as T`
                let src_var_is_numeric = matches!(&e.kind, ExprKind::Var(v) if {
                    let vt = self.var_types.get(v.as_str());
                    matches!(vt, Some(Type::Int | Type::Uint | Type::Float))
                    || matches!(vt, Some(Type::Named(n)) if matches!(n.as_str(), "int" | "uint" | "float" | "i64" | "u64" | "f64" | "usize" | "i32" | "u32" | "f32"))
                });
                if src_var_is_numeric && is_float_ty {
                    return format!("({} as f64)", src);
                }
                if src_var_is_numeric && is_int_ty {
                    return format!("({} as i64)", src);
                }

                // bool → int: direct cast (true=1, false=0), always succeeds
                if src_is_bool_lit && is_int_ty {
                    return format!("({} as i64)", src);
                }
                // int/float literal → float: use `as f64`, not .parse()
                if src_is_numeric_lit && is_float_ty {
                    return format!("({} as f64)", src);
                }
                // numeric literal → bool: always nil (int-to-bool not meaningful in Boring)
                if src_is_numeric_lit && is_bool_ty {
                    return "None".into();
                }
                // Non-string (numeric var) → float/int: use `as T` cast, not .parse()
                if src_is_numeric_var && is_float_ty {
                    return format!("({} as f64)", src);
                }
                if src_is_numeric_var && is_int_ty {
                    return format!("({} as i64)", src);
                }
                // Non-string (numeric var) → bool: None (invalid cast)
                if src_is_numeric_var && is_bool_ty {
                    return "None".into();
                }

                // T as T'actor → Rc::new(RefCell::new(src)) in single-thread mode,
                // Arc::new(Mutex::new(src)) in multi-thread mode.
                // This handles Boring `let x = Struct(...) as Struct'actor` patterns.
                if matches!(ty, crate::ast::Type::Qualified(_, crate::ast::OwnerQual::Actor)) {
                    return match self.config.threading {
                        crate::transpiler::ThreadingMode::Single =>
                            format!("Rc::new(RefCell::new({}))", src),
                        crate::transpiler::ThreadingMode::Multi =>
                            format!("Arc::new(Mutex::new({}))", src),
                    };
                }

                match ty {
                    // string → int/uint/float: use `?` only inside an explicit `try:` body,
                    // never just because the enclosing function returns Result.
                    // This keeps `"42" as int` producing Option<i64> in normal code.
                    Type::Int => if self.in_try_body {
                        format!("{}.trim().parse::<i64>()?", src)
                    } else {
                        format!("{}.trim().parse::<i64>().ok()", src)
                    },
                    Type::Uint => if self.in_try_body {
                        format!("{}.trim().parse::<u64>()?", src)
                    } else {
                        format!("{}.trim().parse::<u64>().ok()", src)
                    },
                    Type::Float => if self.in_try_body {
                        format!("{}.trim().parse::<f64>()?", src)
                    } else {
                        format!("{}.trim().parse::<f64>().ok()", src)
                    },
                    Type::Named(n) if n == "int" => if self.in_try_body {
                        format!("{}.trim().parse::<i64>()?", src)
                    } else {
                        format!("{}.trim().parse::<i64>().ok()", src)
                    },
                    Type::Named(n) if n == "uint" => if self.in_try_body {
                        format!("{}.trim().parse::<u64>()?", src)
                    } else {
                        format!("{}.trim().parse::<u64>().ok()", src)
                    },
                    Type::Named(n) if n == "float" => if self.in_try_body {
                        format!("{}.trim().parse::<f64>()?", src)
                    } else {
                        format!("{}.trim().parse::<f64>().ok()", src)
                    },
                    // string → bool: equality check
                    Type::Bool => format!("({} == \"true\")", src),
                    Type::Named(n) if n == "bool" => format!("({} == \"true\")", src),
                    // numeric/value → string
                    Type::Str => format!("Arc::<str>::from({}.to_string())", src),
                    Type::Named(n) if n == "string" => format!("Arc::<str>::from({}.to_string())", src),
                    // everything else: primitive Rust cast
                    _ => format!("({} as {})", src, dst),
                }
            }

            ExprKind::OptionalField(obj, field) => {
                // Use .clone() so the result is Option<T> (owned), not Option<&T>.
                // This makes nil-coalescing (`?.field else default`) work with matching types.
                let obj_s = self.emit_expr(obj);
                let shadow_name = if let ExprKind::Var(v) = &obj.kind {
                    self.managed_param_shadows.get(v.as_str()).cloned()
                } else { None };
                let is_managed_mutex = shadow_name.is_none() && if let ExprKind::Var(v) = &obj.kind {
                    self.managed_mutex_vars.contains(v.as_str())
                } else { false };
                let is_managed_refcell = if let ExprKind::Var(v) = &obj.kind {
                    self.managed_refcell_vars.contains(v.as_str())
                } else { false };
                if let Some(shadow) = shadow_name {
                    format!("{}.as_ref().map(|__v| __v.{}.clone())", shadow, field)
                } else if is_managed_mutex {
                    // Arc<std::sync::Mutex<T>> — use .lock().unwrap()
                    format!("{}.as_ref().map(|__v| __v.lock().unwrap().{}.clone())", obj_s, field)
                } else if is_managed_refcell {
                    // RefCell<T> — use .borrow()
                    format!("{}.as_ref().map(|__v| __v.borrow().{}.clone())", obj_s, field)
                } else {
                    format!("{}.as_ref().map(|__v| __v.{}.clone())", obj_s, field)
                }
            }
            ExprKind::OptionalMethodCall(obj, method, args) => {
                // Use emit_expr_owned so string literals are coerced to Arc<str> (not &str).
                // Without this, opt?.push("hello") would pass &str where Arc<str> is expected.
                let args_s: Vec<String> = args.iter().map(|a| self.emit_expr_owned(&a.value)).collect();
                let obj_s = self.emit_expr(obj);
                let shadow_name_mc = if let ExprKind::Var(v) = &obj.kind {
                    self.managed_param_shadows.get(v.as_str()).cloned()
                } else { None };
                let is_managed_mutex = shadow_name_mc.is_none() && if let ExprKind::Var(v) = &obj.kind {
                    self.managed_mutex_vars.contains(v.as_str())
                } else { false };
                let is_managed_refcell = if let ExprKind::Var(v) = &obj.kind {
                    self.managed_refcell_vars.contains(v.as_str())
                } else { false };
                if let Some(shadow) = shadow_name_mc {
                    format!("{}.clone().map(|mut __v| __v.{}({}))", shadow, method, args_s.join(", "))
                } else if is_managed_mutex {
                    // Arc<std::sync::Mutex<T>> — use .lock().unwrap()
                    format!("{}.clone().map(|__v| __v.lock().unwrap().{}({}))", obj_s, method, args_s.join(", "))
                } else if is_managed_refcell {
                    // RefCell<T> — use .borrow_mut() for method calls
                    format!("{}.clone().map(|__v| __v.borrow_mut().{}({}))", obj_s, method, args_s.join(", "))
                } else {
                    // Use .clone().map(|mut __v| ...) so that &mut self methods can be called.
                    // Cloning Option<Box<T>> gives an owned value, and `mut __v` allows &mut deref.
                    format!("{}.clone().map(|mut __v| __v.{}({}))", obj_s, method, args_s.join(", "))
                }
            }

            ExprKind::Closure(params, _ret_ty, body, throws, task) => {
                let ps: Vec<String> = params.iter().map(|p| {
                    let name = if p.mutable { format!("mut {}", p.name) } else { p.name.clone() };
                    if let Some(ty) = &p.ty {
                        format!("{}: {}", name, self.emit_type(ty))
                    } else {
                        name
                    }
                }).collect();
                // Emit the closure body with params registered as known locals so that
                // `param.method()` doesn't get misread as a module path `param::method()`.
                let mut sub = self.make_sub();
                for p in params.iter() {
                    sub.known_local_vars.insert(p.name.clone());
                    // Remove any outer-scope var_struct_types entry for this param name.
                    // Without this, a closure param `p` would inherit the type of an outer
                    // variable named `p` (e.g. `p: Parrot`), causing field accesses like
                    // `p.name` to be incorrectly emitted as getter calls `p.name()`.
                    sub.var_struct_types.remove(&p.name);
                }
                // `task` closures: wrap body in `async move { ... }` so they return a Future.
                // `throws` closures: wrap return value in Ok(...).
                if *task {
                    // When the body is a bare `task: ...` expression whose body references Arc
                    // variables from the outer scope, those Arcs would be moved by `async move`.
                    // This is fine for a FnOnce closure but breaks FnMut (e.g. `.map(|x| ...)`)
                    // because the same Arc can't be moved on every call.
                    //
                    // Fix: detect Arc captures in the task body and pre-clone them at the start
                    // of the sync closure body so each invocation creates fresh owned clones
                    // before the `async move` takes them.
                    let param_names: std::collections::HashSet<&str> =
                        params.iter().map(|p| p.name.as_str()).collect();
                    // Collect Arc captures from the task body — works for both:
                    //   ClosureBody::Expr(ExprKind::Task(inner))       — single-line `(x): task: expr`
                    //   ClosureBody::Block([Stmt::Expr(ExprKind::Task(inner)), ...])  — multiline
                    let task_inner_expr: Option<&Expr> = match body {
                        ClosureBody::Expr(e) => {
                            if let ExprKind::Task(inner) = &e.kind { Some(inner.as_ref()) } else { None }
                        }
                        ClosureBody::Block(stmts) => {
                            // Last statement may be the task expression.
                            stmts.last().and_then(|s| match s {
                                Stmt::Expr(e) | Stmt::Return(ReturnStmt { value: Some(e), .. }) => {
                                    if let ExprKind::Task(inner) = &e.kind { Some(inner.as_ref()) } else { None }
                                }
                                _ => None,
                            })
                        }
                    };
                    let pre_clones: String = if let Some(inner) = task_inner_expr {
                        let captured = collect_var_names(inner);
                        let arc_caps: Vec<String> = captured.iter()
                            .filter(|v| {
                                (sub.arc_vars.contains(*v) || sub.string_arc_vars.contains(*v))
                                    && !param_names.contains(v.as_str())
                            })
                            .map(|v| {
                                if sub.rc_vars.contains(v.as_str()) {
                                    format!("let {} = Rc::clone(&{});", v, v)
                                } else {
                                    format!("let {} = Arc::clone(&{});", v, v)
                                }
                            })
                            .collect();
                        arc_caps.join(" ")
                    } else {
                        String::new()
                    };

                    let body_s = match body {
                        ClosureBody::Expr(e) => {
                            let val = sub.emit_expr(e);
                            if *throws { format!("Ok({})", val) } else { val }
                        }
                        ClosureBody::Block(stmts) => {
                            sub.fn_returns_void = false;
                            sub.in_throws = *throws;
                            let n = stmts.len();
                            let inner: Vec<String> = stmts.iter().enumerate().map(|(i, s)| {
                                if i + 1 == n {
                                    sub.emit_stmt_inline(s)
                                } else {
                                    format!("{};", sub.emit_stmt_inline(s))
                                }
                            }).collect();
                            inner.join(" ")
                        }
                    };
                    if pre_clones.is_empty() {
                        return format!("|{}| async move {{ {} }}", ps.join(", "), body_s);
                    } else {
                        // Pre-clone Arcs in a sync wrapper block, then return the async future.
                        return format!("|{}| {{ {} async move {{ {} }} }}", ps.join(", "), pre_clones, body_s);
                    }
                }
                match body {
                    ClosureBody::Expr(e) => {
                        let val = sub.emit_expr(e);
                        if *throws {
                            format!("|{}| Ok({})", ps.join(", "), val)
                        } else {
                            format!("|{}| {}", ps.join(", "), val)
                        }
                    }
                    ClosureBody::Block(stmts) => {
                        // Closure blocks: the last statement should be a value expression.
                        // Clear in_throws and fn_returns_void so if/match branches emit
                        // values without Ok()-wrapping or trailing semicolons.
                        sub.fn_returns_void = false;
                        sub.in_throws = false;
                        let n = stmts.len();
                        let inner: Vec<String> = stmts.iter().enumerate().map(|(i, s)| {
                            if i + 1 == n {
                                // Last stmt: emit as value (if/match need is_last=true).
                                match s {
                                    Stmt::If(if_s) => {
                                        let prev_out = std::mem::take(&mut sub.out);
                                        sub.emit_if(if_s, true);
                                        let result = std::mem::replace(&mut sub.out, prev_out);
                                        result.trim_end_matches('\n').to_string()
                                    }
                                    Stmt::Match(m_s) => {
                                        let prev_out = std::mem::take(&mut sub.out);
                                        sub.emit_match(m_s, true);
                                        let result = std::mem::replace(&mut sub.out, prev_out);
                                        result.trim_end_matches('\n').to_string()
                                    }
                                    _ => sub.emit_stmt_inline(s),
                                }
                            } else {
                                let v = sub.emit_stmt_inline(s);
                                format!("{};", v)
                            }
                        }).collect();
                        format!("|{}| {{ {} }}", ps.join(", "), inner.join(" "))
                    }
                }
            }

            ExprKind::If(s) => {
                // If-as-expression: branches must return values (no semicolons on last stmt).
                // Keep in_throws from the outer context so that `?` is propagated on throws
                // calls inside branches, but set suppress_ok_wrap so the last expression is
                // NOT wrapped in Ok() (it's an expression value, not a function return).
                // If-as-expression: clear fn_return_ty so branch bodies don't inherit the
                // outer function's Optional return type. The let-binding context (emit_let_value)
                // handles Optional coercion at the outer level. For function returns, the
                // if-expression itself is detected as already-Optional by the branch_ends_optional
                // check in emit_stmt.
                let mut sub = self.make_sub();
                sub.fn_returns_void = false;
                sub.suppress_ok_wrap = true; // prevent Ok() wrapping in branch bodies
                sub.fn_return_ty = None; // prevent spurious Some() wrapping in branch bodies
                for (i, (cond, body)) in s.branches.iter().enumerate() {
                    let kw = if i == 0 { "if" } else { "} else if" };
                    let cond_s = sub.emit_expr(cond);
                    sub.line(&format!("{} {} {{", kw, cond_s));
                    sub.indent += 1;
                    sub.emit_body(body);
                    sub.indent -= 1;
                }
                if let Some(else_body) = &s.else_body {
                    sub.line("} else {");
                    sub.indent += 1;
                    sub.emit_body(else_body);
                    sub.indent -= 1;
                }
                sub.line("}");
                format!("{{\n{}}}", sub.out)
            }
            ExprKind::Match(s) => {
                let mut sub = self.make_sub();
                // Match used as an expression — arms must return values, never add `;`.
                sub.fn_returns_void = false;
                sub.suppress_ok_wrap = true; // prevent Ok() wrapping; keep ?-propagation
                sub.fn_return_ty = None; // prevent spurious Some() wrapping in arm bodies
                sub.emit_match(s, true);
                sub.out.trim_end().to_string()
            }
            ExprKind::Block(stmts) => {
                let inner: Vec<String> = stmts.iter().map(|s| self.emit_stmt_inline(s)).collect();
                format!("{{ {} }}", inner.join(" "))
            }
            ExprKind::Do(stmts) => {
                // `do:` block — emit as a proper block using a sub-emitter so that
                // complex statements (for loops, if, etc.) are rendered correctly.
                // Do not inherit `in_throws` from the parent: the block's last expression
                // is not a Result — it's the block's value, not a function return.
                let mut sub = self.make_sub();
                sub.in_throws = false;
                sub.emit_body(stmts);
                format!("{{\n{}}}", sub.out)
            }
            ExprKind::Loop(s) => {
                // Use a sub-emitter so each statement in the body gets proper semicolons/formatting.
                let mut sub = self.make_sub();
                sub.emit_loop(s);
                sub.out.trim_end().to_string()
            }
            ExprKind::TaskWithTimeout(dur_expr, body_expr) => {
                // task(duration): body
                //
                // Emits tokio::time::timeout(dur, async move { body }) in a spawn or inline.
                // The Elapsed error propagates as a Box<dyn Error> catchable by:
                //   • try task(dur): body  else: …   (always works)
                //   • catch:                          (untyped catch-all)
                //
                // Body is built identically to plain Task — Arc vars are cloned into the closure.
                // Resolve leading-dot: `.fromSecs(5)` → `Duration::from_secs(5)`.
                let is_instant_dur = expr_is_instant(dur_expr, &self.instant_vars);
                let dur_type_prefix = if is_instant_dur { "Instant" } else { "Duration" };
                let dur_s = self.resolve_dot_with_type(dur_expr, dur_type_prefix)
                    .unwrap_or_else(|| self.emit_expr(dur_expr));
                let captured = collect_var_names(body_expr);
                let arc_captures: Vec<&str> = captured.iter()
                    .filter(|v| self.arc_vars.contains(*v))
                    .map(String::as_str)
                    .collect();

                let inner_s = if let ExprKind::Block(stmts) = &body_expr.kind {
                    let mut sub = self.make_sub();
                    sub.in_async = true;
                    sub.in_throws = false;
                    sub.emit_body(stmts);
                    format!("{{\n{}}}", sub.out)
                } else {
                    let mut sub = self.make_sub();
                    sub.in_async = true;
                    sub.in_throws = false;
                    format!("{{ {} }}", sub.emit_expr(body_expr))
                };

                let clone_prefix = if arc_captures.is_empty() {
                    String::new()
                } else {
                    arc_captures.iter()
                        .map(|v| {
                            if self.rc_vars.contains(*v) {
                                format!("let {} = Rc::clone(&{});", v, v)
                            } else {
                                format!("let {} = Arc::clone(&{});", v, v)
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ") + " "
                };

                // tokio::time::timeout wraps the body future; Elapsed propagates via ?
                let timeout_fn = if expr_is_instant(dur_expr, &self.instant_vars) {
                    "timeout_at"
                } else {
                    "timeout"
                };
                let timeout_future = format!(
                    "{}async move {{ tokio::time::{}({}, async move {}).await? }}",
                    clone_prefix, timeout_fn, dur_s, inner_s
                );

                // Spawn if inside an async context (produces a JoinHandle),
                // otherwise emit as an inline future expression.
                if self.in_async {
                    // Mark the spawn as a throws JoinHandle so .value uses the ? unwrap
                    let spawn_fn = match self.config.threading {
                        crate::transpiler::ThreadingMode::Single => "tokio::task::spawn_local",
                        crate::transpiler::ThreadingMode::Multi  => "tokio::spawn",
                    };
                    format!("{}({})", spawn_fn, timeout_future)
                } else {
                    timeout_future
                }
            }

            ExprKind::Task(e) => {
                // Auto-detect whether to use tokio::spawn (async) or
                // tokio::task::spawn_blocking (sync/CPU-bound):
                //   task asyncFn(args)  — asyncFn ∈ task_fns  → tokio::spawn
                //   task syncFn(args)   — syncFn ∉ task_fns   → spawn_blocking
                //   task: { async body }                       → tokio::spawn
                //   task: { sync body }  (no await/task)       → spawn_blocking
                let blocking = is_blocking_spawn(e, &self.task_fns);

                // Arc<T> variables captured by the task body must be cloned so the outer
                // binding remains valid after the spawn (tokio::spawn moves its captures).
                let captured = collect_var_names(e);
                let arc_captures: Vec<&str> = captured.iter()
                    .filter(|v| self.arc_vars.contains(*v))
                    .map(String::as_str)
                    .collect();

                // Build the inner body string.
                // For blocking tasks: no `async`, no `.await` on calls (in_async = false).
                // For async tasks:    standard async sub-emitter (in_async = true).
                let inner_s = if let ExprKind::Block(stmts) = &e.kind {
                    let mut sub = self.make_sub();
                    sub.in_async = !blocking;
                    sub.in_throws = false;
                    sub.emit_body(stmts);
                    format!("{{\n{}}}", sub.out)
                } else {
                    let mut sub = self.make_sub();
                    sub.in_async = !blocking;
                    sub.in_throws = false;
                    format!("{{ {} }}", sub.emit_expr(e))
                };

                let clones: String = arc_captures.iter()
                    .map(|v| {
                        if self.rc_vars.contains(*v) {
                            format!("let {} = Rc::clone(&{});", v, v)
                        } else {
                            format!("let {} = Arc::clone(&{});", v, v)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");

                // !Send warning: spawn_local captures Rc vars in single-thread mode.
                if !blocking && matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                    for v in &arc_captures {
                        if self.rc_vars.contains(*v) {
                            eprintln!(
                                "warning: `spawn_local` captures `{}` which is Rc<T> (a !Send type); \
                                 Rc values cannot be sent across task boundaries",
                                v
                            );
                        }
                    }
                }

                if blocking {
                    // Synchronous closure — tokio::task::spawn_blocking(move || { body })
                    let closure = if arc_captures.is_empty() {
                        format!("move || {}", inner_s)
                    } else {
                        format!("{{ {} move || {} }}", clones, inner_s)
                    };
                    format!("tokio::task::spawn_blocking({})", closure)
                } else {
                    // Asynchronous closure — spawn_local (single) or tokio::spawn (multi)
                    let spawn_fn = match self.config.threading {
                        crate::transpiler::ThreadingMode::Single => "tokio::task::spawn_local",
                        crate::transpiler::ThreadingMode::Multi  => "tokio::spawn",
                    };
                    if arc_captures.is_empty() {
                        if self.in_async {
                            format!("{}(async move {})", spawn_fn, inner_s)
                        } else {
                            format!("async move {}", inner_s)
                        }
                    } else {
                        if self.in_async {
                            format!("{}({{ {} async move {} }})", spawn_fn, clones, inner_s)
                        } else {
                            format!("{{ {} async move {} }}", clones, inner_s)
                        }
                    }
                }
            }
            ExprKind::JoinAll(handles) => {
                // Standalone `join [f1, f2]` — emit tokio::join! directly
                let exprs: Vec<String> = handles.iter().map(|e| self.emit_expr(e)).collect();
                format!("tokio::join!({})", exprs.join(", "))
            }
            ExprKind::MacroCall { name, args } => self.emit_macro(name, args),
        }
    }

    pub(crate) fn emit_call(&self, callee: &Expr, args: &[Arg]) -> String {
        if let ExprKind::Var(name) = &callee.kind {
            // Type constructors (PascalCase) — emit as struct literal or ::new()
            let is_type = name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
            if is_type {
                return self.emit_constructor(name, args);
            }
            // Built-in async primitives take priority over fn_sigs dispatch.
            // `wait` / `sleep` — emit as tokio::time::sleep (or sleep_until for Instant).
            // Must come before fn_sigs check because wait/timeout are now in fn_sigs
            // (so that DotIdent hints work), but they need to emit tokio:: paths, not
            // a plain `wait(...)` call.
            if (name == "sleep" || name == "wait") && args.len() == 1 && self.in_async {
                let is_instant = expr_is_instant(&args[0].value, &self.instant_vars);
                let type_prefix = if is_instant { "Instant" } else { "Duration" };
                // Resolve leading-dot static call: `.fromSecs(n)` → `Duration::from_secs(n)`
                let arg = self.resolve_dot_with_type(&args[0].value, type_prefix)
                    .unwrap_or_else(|| self.emit_expr(&args[0].value));
                return if is_instant {
                    format!("tokio::time::sleep_until({}).await", arg)
                } else {
                    format!("tokio::time::sleep({}).await", arg)
                };
            }
            // Overloaded function call — select the right mangled name based on arg types.
            if self.overloaded_fn_names.contains(name.as_str()) {
                let overloads = self.fn_overload_decls.get(name.as_str())
                    .cloned()
                    .unwrap_or_default();
                // Try to find a matching overload by type inference.
                let chosen = overloads.iter().find(|decl| {
                    if decl.params.len() != args.len() { return false; }
                    decl.params.iter().zip(args.iter()).all(|(param, arg)| {
                        match &param.ty {
                            None => true,
                            Some(expected_ty) => {
                                let inferred = infer_overload_expr_type(
                                    &arg.value,
                                    &self.var_types,
                                    &self.fn_return_types,
                                    &self.struct_fields,
                                );
                                match inferred {
                                    Some(inferred_ty) => types_compatible(expected_ty, &inferred_ty),
                                    None => true, // can't determine — optimistically match
                                }
                            }
                        }
                    })
                }).or_else(|| overloads.first());

                if let Some(decl) = chosen {
                    let mangled = mangle_overload_name(name, &decl.params);
                    let args_s = self.emit_args_coerced(&mangled, args);
                    let base = format!("{}({})", mangled, args_s);
                    let is_task = self.in_async && self.task_fns.contains(name.as_str());
                    let propagates = (self.in_try_body || self.in_throws) && self.fn_throws.contains(name.as_str());
                    return match (is_task, propagates) {
                        (true,  true)  => format!("{}.await?", base),
                        (true,  false) => format!("{}.await",  base),
                        (false, true)  => format!("{}?",       base),
                        (false, false) => base,
                    };
                }
            }
            // User-defined functions (and stdlib functions registered in fn_sigs).
            if self.fn_sigs.contains_key(name.as_str()) {
                let args_s = self.emit_args_coerced(name, args);
                let base = format!("{}({})", escape_rust_keyword(name), args_s);
                let is_task = self.in_async
                    && self.task_fns.contains(name.as_str())
                    && !self.stream_fns.contains(name.as_str());
                let propagates = (self.in_try_body || self.in_throws) && self.fn_throws.contains(name.as_str());
                // Correct ordering: async task calls must be `.await` then `?` (not `?` then `.await`).
                return match (is_task, propagates) {
                    (true,  true)  => format!("{}.await?", base),
                    (true,  false) => format!("{}.await",  base),
                    (false, true)  => format!("{}?",       base),
                    (false, false) => base,
                };
            }
            // Special case: `timeout(dur, future_expr)` — the second arg must be a future.
            // `tokio::time::timeout` takes `F: Future`, so the second arg should be the future
            // expression directly (e.g. `tokio::time::sleep(dur)` or a closure async block).
            // When the second arg is a `task expr` (Task node), emit the inner expression
            // directly as a future rather than spawning. This avoids the JoinHandle type mismatch.
            if name == "timeout" && args.len() == 2 && self.in_async {
                // Resolve leading-dot syntax for the duration/deadline argument:
                //   timeout(.fromSecs(5), …)  →  timeout(Duration::from_secs(5), …)
                let is_instant = expr_is_instant(&args[0].value, &self.instant_vars);
                let type_prefix = if is_instant { "Instant" } else { "Duration" };
                let dur = self.resolve_dot_with_type(&args[0].value, type_prefix)
                    .unwrap_or_else(|| self.emit_expr(&args[0].value));
                // For `task inner` args: emit the inner expression directly (already a Future).
                // For async method calls that end with `.await`, wrap in `async move { }` so
                // timeout receives a `Future<Output=T>` rather than the already-awaited value.
                // For plain futures (e.g. tokio::time::sleep): pass through as-is.
                let future_expr = {
                    // `timeout` needs a Future<Output=T>, not an already-awaited value.
                    // Three forms for the second argument:
                    //   task f(args)       — TaskExpr: emit inner expression directly as future
                    //   f                  — Callable<T> (task fn ref): call it as f() to get future
                    //   <already a future> — strip any trailing .await added by the expression emitter
                    let raw = match &args[1].value.kind {
                        ExprKind::Task(inner_e) => self.emit_expr(inner_e),
                        // Bare variable: check if it's a task function — call it to get the future
                        ExprKind::Var(fn_name)
                            if self.task_fns.contains(fn_name.as_str())
                               || self.fn_sigs.contains_key(fn_name.as_str()) =>
                        {
                            // If it's a known task_fn with no args: call it to produce the future
                            if self.task_fns.contains(fn_name.as_str()) {
                                format!("{}()", fn_name)
                            } else {
                                self.emit_expr(&args[1].value)
                            }
                        }
                        // Zero-arg trailing closure `(): body` or `(): fetch()` —
                        // unwrap the body and emit it directly as the future expression.
                        // This handles `timeout(dur): fetch()` → future is `fetch()`, not `|| fetch()`.
                        ExprKind::Closure(params, _, body, _, _) if params.is_empty() => {
                            match body {
                                ClosureBody::Expr(e) => self.emit_expr(e),
                                ClosureBody::Block(stmts) => {
                                    let mut sub = self.make_sub();
                                    sub.in_async = true;
                                    sub.in_throws = false;
                                    sub.emit_body(stmts);
                                    format!("async move {{{}}}", sub.out)
                                }
                            }
                        }
                        _ => self.emit_expr(&args[1].value),
                    };
                    if let Some(stripped) = raw.strip_suffix(".await") {
                        stripped.to_string()
                    } else if let Some(stripped) = raw.strip_suffix(".await?") {
                        stripped.to_string()
                    } else {
                        raw
                    }
                };
                // In a cancellable function: use select! to race future vs timer vs cancel.
                if self.in_cancellable_fn {
                    let timer_fn = if expr_is_instant(&args[0].value, &self.instant_vars) {
                        format!("tokio::time::sleep_until({})", dur)
                    } else {
                        format!("tokio::time::sleep({})", dur)
                    };
                    return if self.in_throws || self.in_try_body {
                        format!(
                            "{{ tokio::select! {{ __boring_r = ({}) => Ok(__boring_r), _ = {} => Err(Box::new(BoringError::Other(std::any::TypeId::of::<Error>(), Box::new(Error::Expired) as Box<dyn BoringVal + Send + Sync>))), _ = __task_cancel.cancelled() => Err(Box::new(BoringError::Other(std::any::TypeId::of::<Error>(), Box::new(Error::Cancelled) as Box<dyn BoringVal + Send + Sync>))), }} }}?",
                            future_expr, timer_fn
                        )
                    } else {
                        format!(
                            "{{ tokio::select! {{ __boring_r = ({}) => Some(__boring_r), _ = {} => None, _ = __task_cancel.cancelled() => None, }} }}",
                            future_expr, timer_fn
                        )
                    };
                }
                // Always add .await — TryElse clears in_throws/in_try_body to avoid adding `?`.
                let base = if expr_is_instant(&args[0].value, &self.instant_vars) {
                    format!("tokio::time::timeout_at({}, {}).await", dur, future_expr)
                } else {
                    format!("tokio::time::timeout({}, {}).await", dur, future_expr)
                };
                // In throws/try context, propagate Elapsed errors with `?`.
                return if self.in_throws || self.in_try_body {
                    format!("{}?", base)
                } else {
                    base
                };
            }
            // Task fn params: calling them produces a future that needs .await.
            // When the param type is also `throws` (returns Future<Output=Result<T,_>>),
            // add `?` in a throws / try context so errors propagate correctly.
            if self.in_async && self.task_vars.contains(name.as_str()) {
                let args_s = self.emit_args(args);
                let call_s = format!("{}({})", escape_rust_keyword(name), args_s);
                return if self.in_throws || self.in_try_body {
                    format!("{}.await?", call_s)
                } else {
                    format!("{}.await", call_s)
                };
            }
            // Non-task fn params declared as `throws` return Result — add `?` in throws context.
            if (self.in_throws || self.in_try_body) && self.throws_fn_params.contains(name.as_str()) {
                let args_s = self.emit_args(args);
                let call_s = format!("{}({})", escape_rust_keyword(name), args_s);
                return format!("{}?", call_s);
            }
            return self.emit_builtin_call(name, args);
        }
        // Enum variant constructor: Value.NativeFn(name, val) → Value::NativeFn(name, Box::new(val))
        // for recursive fields. Check if callee is Field(Var(EnumType), VariantName).
        if let ExprKind::Field(obj_expr, variant_name) = &callee.kind {
            if let ExprKind::Var(enum_type) = &obj_expr.kind {
                let key = format!("{}::{}", enum_type, variant_name);
                if let Some(field_types) = self.enum_variant_field_types.get(&key).cloned() {
                    let callee_s = format!("{}::{}", enum_type, variant_name);
                    let args_s: Vec<String> = args.iter().enumerate().map(|(i, a)| {
                        let raw = self.emit_let_value(field_types.get(i), &a.value);
                        let rec_key = format!("{}::{}::{}", enum_type, variant_name, i);
                        if self.recursive_fields.contains(&rec_key) {
                            if matches!(field_types.get(i), Some(Type::Optional(_))) {
                                format!("{}.map(Box::new)", raw)
                            } else {
                                format!("Box::new({})", raw)
                            }
                        } else {
                            raw
                        }
                    }).collect();
                    return format!("{}({})", callee_s, args_s.join(", "));
                }
            }
        }
        let callee_s = self.emit_expr(callee);
        let args_s = self.emit_args(args);
        format!("{}({})", callee_s, args_s)
    }

    pub(crate) fn emit_generic_call(&self, callee: &Expr, type_args: &[Type], args: &[Arg]) -> String {
        if let ExprKind::Var(name) = &callee.kind {
            match name.as_str() {
                "channel" => {
                    let ty = type_args.first()
                        .map(|t| self.emit_type(t))
                        .unwrap_or_else(|| "_".to_string());
                    // Capacity: second type arg (channel<T, 32>) or first call arg (channel<T>(cap)).
                    let cap = if type_args.len() >= 2 {
                        match &type_args[1] {
                            crate::ast::Type::Named(n) => n.clone(),
                            other => self.emit_type(other),
                        }
                    } else {
                        args.first()
                            .map(|a| self.emit_expr(&a.value))
                            .unwrap_or_else(|| "0".to_string())
                    };
                    let channel_mod = match self.config.threading {
                        crate::transpiler::ThreadingMode::Single => {
                            self.uses_local_channel.set(true);
                            "local_channel::mpsc"
                        }
                        crate::transpiler::ThreadingMode::Multi  => "tokio::sync::mpsc",
                    };
                    // local_channel::mpsc::channel() is unbounded — no capacity argument.
                    if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                        return format!("{}::channel::<{}>()", channel_mod, ty);
                    }
                    return format!("{}::channel::<{}>({})", channel_mod, ty, cap);
                }
                "oneshot" => {
                    let ty = type_args.first()
                        .map(|t| self.emit_type(t))
                        .unwrap_or_else(|| "_".to_string());
                    return format!("tokio::sync::oneshot::channel::<{}>()", ty);
                }
                "broadcast" => {
                    let ty = type_args.first()
                        .map(|t| self.emit_type(t))
                        .unwrap_or_else(|| "_".to_string());
                    if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                        self.uses_local_broadcast.set(true);
                        return format!("local_broadcast::<{}>()", ty);
                    }
                    let cap = if type_args.len() >= 2 {
                        match &type_args[1] {
                            crate::ast::Type::Named(n) => n.clone(),
                            other => self.emit_type(other),
                        }
                    } else {
                        args.first()
                            .map(|a| self.emit_expr(&a.value))
                            .unwrap_or_else(|| "16".to_string())
                    };
                    return format!("tokio::sync::broadcast::channel::<{}>({})", ty, cap);
                }
                "watch" => {
                    let ty = type_args.first()
                        .map(|t| self.emit_type(t))
                        .unwrap_or_else(|| "_".to_string());
                    let init = args.first()
                        .map(|a| self.emit_expr(&a.value))
                        .unwrap_or_else(|| "Default::default()".to_string());
                    return format!("tokio::sync::watch::channel::<{}>({})", ty, init);
                }
                "timeout" => {
                    // timeout(dur, fut) — contextual:
                    //   cancellable fn  → select! racing future / sleep / cancel token
                    //   throws context  → .await?     (Elapsed propagated as error)
                    //   otherwise       → .await.ok() (returns T?)
                    let dur = args.first()
                        .map(|a| self.emit_expr(&a.value))
                        .unwrap_or_else(|| "Duration::from_secs(0)".to_string());
                    let raw_fut = args.get(1)
                        .map(|a| self.emit_expr(&a.value))
                        .unwrap_or_else(|| "async {}".to_string());
                    // Strip trailing .await so we pass a Future, not its value.
                    let fut = raw_fut.strip_suffix(".await")
                        .or_else(|| raw_fut.strip_suffix(".await?"))
                        .unwrap_or(&raw_fut)
                        .to_string();
                    if self.in_cancellable_fn {
                        return if self.in_throws || self.in_try_body {
                            format!(
                                "{{ tokio::select! {{ __boring_r = ({}) => Ok(__boring_r), _ = tokio::time::sleep({}) => Err(Box::new(BoringError::Other(std::any::TypeId::of::<Error>(), Box::new(Error::Expired) as Box<dyn BoringVal + Send + Sync>))), _ = __task_cancel.cancelled() => Err(Box::new(BoringError::Other(std::any::TypeId::of::<Error>(), Box::new(Error::Cancelled) as Box<dyn BoringVal + Send + Sync>))), }} }}?",
                                fut, dur
                            )
                        } else {
                            format!(
                                "{{ tokio::select! {{ __boring_r = ({}) => Some(__boring_r), _ = tokio::time::sleep({}) => None, _ = __task_cancel.cancelled() => None, }} }}",
                                fut, dur
                            )
                        };
                    }
                    let base = format!("tokio::time::timeout({}, {}).await", dur, fut);
                    return if self.in_throws || self.in_try_body {
                        format!("{}?", base)
                    } else {
                        format!("{}.ok()", base)
                    };
                }
                // from_json<T>(s) → serde_json::from_str::<T>(&s)
                // In a throws/try context propagates the error; otherwise wraps in .ok().
                "fromJson" => {
                    self.uses_serde.set(true);
                    let ty = type_args.first()
                        .map(|t| self.emit_type(t))
                        .unwrap_or_else(|| "_".to_string());
                    let s = args.first()
                        .map(|a| self.emit_expr(&a.value))
                        .unwrap_or_else(|| "\"\"".to_string());
                    let base = format!("serde_json::from_str::<{}>(&{})", ty, s);
                    return if self.in_throws || self.in_try_body {
                        format!("{}?", base)
                    } else {
                        format!("{}.ok()", base)
                    };
                }
                _ => {}
            }
            // Fallback: emit as a regular call, ignore type args
            let ty_args_s: Vec<String> = type_args.iter().map(|t| self.emit_type(t)).collect();
            let args_s: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
            format!("{}::<{}>({})", name, ty_args_s.join(", "), args_s.join(", "))
        } else {
            let callee_s = self.emit_expr(callee);
            let ty_args_s: Vec<String> = type_args.iter().map(|t| self.emit_type(t)).collect();
            let args_s: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
            format!("{}::<{}>({})", callee_s, ty_args_s.join(", "), args_s.join(", "))
        }
    }

    pub(crate) fn emit_pipe(&self, lhs: &Expr, name: &str, args: &[Arg]) -> String {
        // If the name is a known standalone function, insert lhs as first argument.
        // Otherwise treat it as a method call on lhs.
        if self.fn_sigs.contains_key(name) {
            let lhs_s = self.emit_expr(lhs);
            let rest: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
            let all_args = if rest.is_empty() {
                lhs_s
            } else {
                format!("{}, {}", lhs_s, rest.join(", "))
            };
            let base = format!("{}({})", escape_rust_keyword(name), all_args);
            let is_task = self.in_async && self.task_fns.contains(name);
            let propagates = (self.in_try_body || self.in_throws) && self.fn_throws.contains(name);
            match (is_task, propagates) {
                (true,  true)  => format!("{}.await?", base),
                (true,  false) => format!("{}.await",  base),
                (false, true)  => format!("{}?",       base),
                (false, false) => base,
            }
        } else {
            // Method call: delegate directly to emit_method_call with the real lhs expr.
            self.emit_method_call(lhs, name, args)
        }
    }

    pub(crate) fn emit_constructor(&self, name: &str, args: &[Arg]) -> String {
        let result = self.emit_constructor_inner(name, args);
        // Check if the current function returns T' in managed mode → wrap in managed actor.
        if let Some(Type::Qualified(inner, OwnerQual::Owned)) = &self.fn_return_ty {
            if matches!(inner.as_ref(), Type::Named(n) if n == name) {
                if self.is_managed_owned_user(self.fn_return_ty.as_ref().unwrap()) {
                    return match self.config.threading {
                        crate::transpiler::ThreadingMode::Multi =>
                            format!("Arc::new(std::sync::Mutex::new({}))", result),
                        crate::transpiler::ThreadingMode::Single =>
                            format!("RefCell::new({})", result),
                    };
                }
                // Strict mode: wrap in Box<T>
                return format!("Box::new({})", result);
            }
        }
        result
    }

    pub(crate) fn emit_constructor_inner(&self, name: &str, args: &[Arg]) -> String {
        // Result constructors: `Ok(v)` / `Err(e)` are Rust built-ins, not struct types.
        if name == "Ok" || name == "Err" {
            let args_s = self.emit_args(args);
            return format!("{}({})", name, args_s);
        }
        // Non-fn type alias resolving to a qualified type: construct via the alias.
        // e.g. `AP(3, 4)` where `AP = APoint'` (Box<APoint>) → `Box::new(APoint::new(3, 4))`.
        // e.g. `ANode(99)` where `ANode = ATree'auto` (Rc<ATree>) → `Rc::new(ATree::new(99))`.
        if let Some(resolved) = self.non_fn_type_aliases.get(name) {
            let resolved = resolved.clone();
            match &resolved {
                Type::Qualified(inner, OwnerQual::Owned) => {
                    if let Type::Named(inner_name) = inner.as_ref() {
                        let inner_s = self.emit_constructor_inner(inner_name, args);
                        // Managed mode: wrap in Arc<std::sync::Mutex<T>> or RefCell<T>
                        if self.is_managed_owned_user(&resolved) {
                            return match self.config.threading {
                                crate::transpiler::ThreadingMode::Multi =>
                                    format!("Arc::new(std::sync::Mutex::new({}))", inner_s),
                                crate::transpiler::ThreadingMode::Single =>
                                    format!("RefCell::new({})", inner_s),
                            };
                        }
                        return format!("Box::new({})", inner_s);
                    }
                }
                Type::Qualified(inner, OwnerQual::Shared) => {
                    if let Type::Named(inner_name) = inner.as_ref() {
                        let inner_s = self.emit_constructor_inner(inner_name, args);
                        return match self.config.threading {
                            crate::transpiler::ThreadingMode::Single => format!("Rc::new({})", inner_s),
                            crate::transpiler::ThreadingMode::Multi  => format!("Arc::new({})", inner_s),
                        };
                    }
                }
                Type::Qualified(inner, OwnerQual::Stack | OwnerQual::Copy) => {
                    if let Type::Named(inner_name) = inner.as_ref() {
                        return self.emit_constructor_inner(inner_name, args);
                    }
                }
                Type::Named(inner_name) => {
                    // Simple named alias (e.g. `ADog2 = ADog`) → emit inner constructor.
                    let inner_name = inner_name.clone();
                    return self.emit_constructor_inner(&inner_name, args);
                }
                _ => {}
            }
        }
        // Newtype wrapper: `UserId(42)` → `UserId(42)` (tuple struct constructor).
        if self.newtype_types.contains(name) {
            let arg_s = if let Some(a) = args.first() {
                // String newtypes have inner type `String`; emit_expr_owned converts
                // string literals from `&str` → `"...".to_string()`.
                let inner = self.newtype_inner.get(name).cloned().unwrap_or_default();
                if inner == "String" {
                    // Newtype inner is String (owned); convert literals directly without Arc.
                    match &a.value.kind {
                        ExprKind::Str(s) => format!("\"{}\".to_string()", escape_str(s)),
                        ExprKind::StringInterp(_) => self.emit_expr(&a.value),
                        _ => {
                            // Variable or expression: may be Arc<str> — unwrap to String.
                            let raw = self.emit_expr(&a.value);
                            format!("(*{}).clone()", raw)
                        }
                    }
                } else {
                    self.emit_expr(&a.value)
                }
            } else {
                "Default::default()".to_string()
            };
            return format!("{}({})", name, arg_s);
        }
        if args.is_empty() {
            // Stdlib collection constructors need turbofish to avoid "type annotations needed".
            match name {
                "HashSet" => return "HashSet::<i64>::new()".into(),
                "HashMap" => return "HashMap::<Arc<str>, i64>::new()".into(),
                _ => {}
            }
            // No-field struct: emit `Struct {}` instead of `Struct::new()`.
            if self.struct_fields.get(name).map(|f| f.is_empty()).unwrap_or(false) {
                return format!("{} {{}}", name);
            }
            // If the struct has an init body, call ::new() — filling in defaults if available.
            if self.struct_has_init_body.contains(name) {
                if let Some(defaults) = self.struct_init_defaults.get(name).cloned() {
                    let def_args: Vec<String> = defaults.iter().filter_map(|d| d.clone()).collect();
                    return format!("{}::new({})", name, def_args.join(", "));
                }
                return format!("{}::new()", name);
            }
            // Struct has fields but no init body — use struct literal with defaults.
            if let Some(fields) = self.struct_fields.get(name).cloned() {
                if !fields.is_empty() {
                    let init_defaults = self.struct_init_defaults.get(name).cloned().unwrap_or_default();
                    let lit_fields: Vec<String> = fields.iter().enumerate()
                        .filter_map(|(i, (fname, fty))| {
                            let key = format!("{}::{}", name, fname);
                            if let Some((is_copy, _, default_val)) = self.transient_fields.get(&key) {
                                let init = if *is_copy {
                                    format!("std::cell::Cell::new({})", default_val)
                                } else {
                                    format!("std::cell::RefCell::new({})", default_val)
                                };
                                Some(format!("{}: {}", fname, init))
                            } else if let Some(Some(def)) = init_defaults.get(i) {
                                Some(format!("{}: {}", fname, def))
                            } else if matches!(fty, Type::Optional(_)) {
                                Some(format!("{}: None", fname))
                            } else {
                                None
                            }
                        })
                        .collect();
                    if lit_fields.len() == fields.len() || !lit_fields.is_empty() {
                        return format!("{} {{ {} }}", name, lit_fields.join(", "));
                    }
                }
            }
            // Enum variant with no fields (unit variant).
            if let Some(enum_name) = self.enum_variants.get(name) {
                return format!("{}::{}", enum_name, name);
            }
            return format!("{}::new()", name);
        }
        // Struct spread: `Name(..base, field = override, ...)` → `Name { field: override, ..base }`
        // Rust struct update syntax requires the `..base` to be last.
        let has_spread = args.iter().any(|a| a.spread);
        if has_spread {
            let spread_exprs: Vec<String> = args.iter()
                .filter(|a| a.spread)
                .map(|a| self.emit_expr(&a.value))
                .collect();
            let labeled_fields: Vec<String> = args.iter()
                .filter(|a| a.label.is_some())
                .map(|a| {
                    let label = a.label.as_ref().unwrap();
                    let field_ty = self.struct_fields.get(name)
                        .and_then(|fs| fs.iter().find(|(n, _)| n == label))
                        .map(|(_, ty)| ty);
                    let val = self.emit_let_value(field_ty, &a.value);
                    format!("{}: {}", label, val)
                })
                .collect();
            // Combine: explicit fields first, then spread bases.
            // Use `.clone()` so that spreading the same base twice doesn't
            // move it on the first use and leave it inaccessible on the second.
            let mut parts = labeled_fields;
            parts.extend(spread_exprs.iter().map(|e| format!("..{}.clone()", e)));
            return format!("{} {{ {} }}", name, parts.join(", "));
        }

        // If all args are labeled (explicit label or closure-style `|field| expr`) → struct literal.
        // In Boring, `Struct(field: expr)` is parsed as a single-param closure |field| expr.
        let all_labeled = args.iter().all(|a| {
            a.label.is_some() || matches!(&a.value.kind, ExprKind::Closure(params, _, _, _, _) if params.len() == 1)
        });
        if all_labeled {
            let mut fields: Vec<String> = args.iter()
                .map(|a| {
                    // Determine the field label: explicit label or closure-style param name.
                    let label: String = if let Some(l) = &a.label {
                        l.clone()
                    } else if let ExprKind::Closure(params, _, _, _, _) = &a.value.kind {
                        params[0].name.clone()
                    } else { unreachable!() };
                    let label = &label;
                    // When arg is a closure-style labeled arg `|field| expr`, unwrap to get the value.
                    let eff_value: &Expr = if let ExprKind::Closure(params, _, body, _, _) = &a.value.kind {
                        if params.len() == 1 {
                            if let ClosureBody::Expr(e) = body { e.as_ref() } else { &a.value }
                        } else { &a.value }
                    } else { &a.value };
                    // Look up declared field type for proper Optional/string coercion
                    let field_ty = self.struct_fields.get(name)
                        .and_then(|fs| fs.iter().find(|(n, _)| n == label))
                        .map(|(_, ty)| ty);
                    let mutex_key = format!("{}::{}", name, label);
                    let val = if self.struct_mutex_fields.contains(&mutex_key) {
                        let inner_ty = field_ty.and_then(Self::mutex_inner);
                        let raw = self.emit_let_value(inner_ty, eff_value);
                        match self.config.threading {
                            crate::transpiler::ThreadingMode::Multi =>
                                format!("Arc::new(tokio::sync::Mutex::new({}))", raw),
                            crate::transpiler::ThreadingMode::Single =>
                                format!("Rc::new(RefCell::new({}))", raw),
                        }
                    } else if self.recursive_fields.contains(&mutex_key) {
                        // Recursive struct field — wrap in Box::new() at construction site.
                        let raw = self.emit_let_value(field_ty, eff_value);
                        if matches!(field_ty, Some(Type::Optional(_))) {
                            format!("{}.map(Box::new)", raw)
                        } else {
                            format!("Box::new({})", raw)
                        }
                    } else {
                        self.emit_let_value(field_ty, eff_value)
                    };
                    format!("{}: {}", label, val)
                })
                .collect();
            // Append transient fields that weren't provided by the user.
            let provided: std::collections::HashSet<String> = args.iter()
                .filter_map(|a| {
                    a.label.clone().or_else(|| {
                        if let ExprKind::Closure(params, _, _, _, _) = &a.value.kind {
                            if params.len() == 1 { Some(params[0].name.clone()) } else { None }
                        } else { None }
                    })
                })
                .collect();
            for (key, (is_copy, _, default_val)) in &self.transient_fields {
                if let Some(field_name) = key.strip_prefix(&format!("{}::", name)) {
                    if !provided.contains(field_name) {
                        let init = if *is_copy {
                            format!("std::cell::Cell::new({})", default_val)
                        } else {
                            format!("std::cell::RefCell::new({})", default_val)
                        };
                        fields.push(format!("{}: {}", field_name, init));
                    }
                }
            }
            // Append var T'task fields missing from the call with a Mutex-wrapped default.
            for key in &self.struct_mutex_fields.clone() {
                if let Some(field_name) = key.strip_prefix(&format!("{}::", name)) {
                    if !provided.contains(field_name) {
                        let init = match self.config.threading {
                            crate::transpiler::ThreadingMode::Multi =>
                                format!("Arc::new(tokio::sync::Mutex::new(Default::default()))"),
                            crate::transpiler::ThreadingMode::Single =>
                                format!("Rc::new(RefCell::new(Default::default()))"),
                        };
                        fields.push(format!("{}: {}", field_name, init));
                    }
                }
            }
            // Append regular optional/T'auto/T'weak fields not provided — default to None.
            if let Some(known_fields) = self.struct_fields.get(name).cloned() {
                for (fname, fty) in &known_fields {
                    if !provided.contains(fname.as_str()) {
                        // Skip transient and mutex fields already handled above.
                        let tkey = format!("{}::{}", name, fname);
                        if self.transient_fields.contains_key(&tkey)
                            || self.struct_mutex_fields.contains(&tkey)
                        {
                            continue;
                        }
                        // Optional fields (including Optional<Qualified<...>>) default to None.
                        if matches!(fty, Type::Optional(_)) {
                            fields.push(format!("{}: None", fname));
                        }
                    }
                }
            }
            format!("{} {{ {} }}", name, fields.join(", "))
        } else {
            // Positional args: if struct fields are known and no explicit new() exists
            // (e.g. generic structs), emit a struct literal using fields in declaration order.
            // Otherwise fall back to ::new(args).

            // If the struct has an init with a body, route to ::new(args) — the body may
            // set computed fields (e.g. `self.area = 3.14 * r * r`) that can't be in a literal.
            if self.struct_has_init_body.contains(name) {
                // Fill in default args for any omitted trailing params.
                let mut all_args: Vec<String> = args.iter()
                    .map(|a| self.emit_expr(&a.value))
                    .collect();
                if let Some(defaults) = self.struct_init_defaults.get(name).cloned() {
                    for i in all_args.len()..defaults.len() {
                        if let Some(def) = &defaults[i] {
                            all_args.push(def.clone());
                        }
                    }
                }
                let args_s = all_args.join(", ");
                return format!("{}::new({})", name, args_s);
            }

            if let Some(fields) = self.struct_fields.get(name) {
                if !fields.is_empty() && args.len() <= fields.len() {
                    // Check if the struct has an init (new() function); if so, use ::new().
                    // Heuristic: if all args are positional and fields are known, use struct literal.
                    let lit_fields: Vec<String> = args.iter().enumerate()
                        .map(|(i, a)| {
                            let (fname, fty) = &fields[i];
                            // `name: expr` in Boring struct call is parsed as a single-param
                            // closure `|name| expr` when `:` is used. If the closure param
                            // matches the field name, unwrap and treat as a labeled value.
                            let effective_value = if let ExprKind::Closure(params, _, body, _, _) = &a.value.kind {
                                if params.len() == 1 && params[0].name == *fname {
                                    match body {
                                        ClosureBody::Expr(e) => e.as_ref(),
                                        _ => &a.value,
                                    }
                                } else { &a.value }
                            } else { &a.value };
                            let rec_key = format!("{}::{}", name, fname);
                            let val = if self.recursive_fields.contains(&rec_key) {
                                let raw = self.emit_let_value(Some(fty), effective_value);
                                if matches!(fty, Type::Optional(_)) {
                                    format!("{}.map(Box::new)", raw)
                                } else {
                                    format!("Box::new({})", raw)
                                }
                            } else {
                                self.emit_let_value(Some(fty), effective_value)
                            };
                            format!("{}: {}", fname, val)
                        })
                        .collect();
                    // Fill missing fields with defaults if any.
                    // Priority: init param defaults > transient Cell defaults > Optional → None.
                    let init_defaults = self.struct_init_defaults.get(name).cloned().unwrap_or_default();
                    let extra_fields: Vec<String> = fields.iter().skip(args.len()).enumerate()
                        .filter_map(|(offset_i, (fname, fty))| {
                            let param_idx = args.len() + offset_i;
                            let key = format!("{}::{}", name, fname);
                            if let Some((is_copy, _, default_val)) = self.transient_fields.get(&key) {
                                let init = if *is_copy {
                                    format!("std::cell::Cell::new({})", default_val)
                                } else {
                                    format!("std::cell::RefCell::new({})", default_val)
                                };
                                Some(format!("{}: {}", fname, init))
                            } else if let Some(Some(def)) = init_defaults.get(param_idx) {
                                // Init param had an explicit default value.
                                Some(format!("{}: {}", fname, def))
                            } else if matches!(fty, Type::Optional(_)) {
                                Some(format!("{}: None", fname))
                            } else {
                                None
                            }
                        })
                        .collect();
                    let mut all_fields = lit_fields;
                    all_fields.extend(extra_fields);
                    return format!("{} {{ {} }}", name, all_fields.join(", "));
                }
            }
            // Enum variant with positional args.
            if let Some(enum_name) = self.enum_variants.get(name) {
                let args_s = self.emit_args(args);
                return format!("{}::{}({})", enum_name, name, args_s);
            }
            // Fallback: call ::new(args) (requires a new() function to exist)
            let args_s = self.emit_args(args);
            // Semaphore::new and similar tokio primitives expect usize, but Boring's
            // `uint` maps to u64. Cast the first argument to usize automatically.
            if matches!(name, "Semaphore" | "RwLock") {
                return format!("{}::new({} as usize)", name, args_s);
            }
            format!("{}::new({})", name, args_s)
        }
    }

    pub(crate) fn emit_builtin_call(&self, name: &str, args: &[Arg]) -> String {
        match name {
            // some(x) → Some(x): wrap a value in Option
            "some" if args.len() == 1 => {
                let v = self.emit_expr_owned(&args[0].value);
                format!("Some({})", v)
            }
            "print" | "println" => {
                self.emit_print_call(true, args)
            }
            "write" | "eprint" => {
                self.emit_print_call(false, args)
            }
            "format" => {
                self.emit_print_call_named("format", args)
            }
            // Log-level builtins: map to the `log` crate macros.
            // Requires `log = "0.4"` in Cargo.toml.
            "error" | "warn" | "info" | "debug" | "trace" => {
                self.uses_log.set(true);
                self.emit_print_call_named(&format!("log::{}", name), args)
            }
            "assert" => {
                if args.len() == 1 {
                    format!("assert!({})", self.emit_expr(&args[0].value))
                } else {
                    let cond = self.emit_expr(&args[0].value);
                    let msg = self.emit_expr(&args[1].value);
                    format!("assert!({}, \"{{:?}}\", {})", cond, msg)
                }
            }
            "assert_eq" => {
                let a = self.emit_expr(&args[0].value);
                let b = self.emit_expr(&args[1].value);
                if args.len() > 2 {
                    let msg = self.emit_expr(&args[2].value);
                    format!("assert_eq!({}, {}, \"{{:?}}\", {})", a, b, msg)
                } else {
                    format!("assert_eq!({}, {})", a, b)
                }
            }
            "assert_neq" => {
                let a = self.emit_expr(&args[0].value);
                let b = self.emit_expr(&args[1].value);
                format!("assert_ne!({}, {})", a, b)
            }
            "panic" => {
                if args.is_empty() {
                    "panic!(\"explicit panic\")".into()
                } else {
                    format!("panic!(\"{{:?}}\", {})", self.emit_expr(&args[0].value))
                }
            }
            "dbg" => {
                if args.is_empty() {
                    "dbg!()".into()
                } else if args.len() == 1 {
                    format!("dbg!({})", self.emit_expr(&args[0].value))
                } else {
                    let args_s: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
                    format!("dbg!({})", args_s.join(", "))
                }
            }
            "todo" => {
                if args.is_empty() {
                    "todo!()".into()
                } else {
                    format!("todo!(\"{{:?}}\", {})", self.emit_expr(&args[0].value))
                }
            }
            "unreachable" => {
                if args.is_empty() {
                    "unreachable!()".into()
                } else {
                    format!("unreachable!(\"{{:?}}\", {})", self.emit_expr(&args[0].value))
                }
            }
            "len" => {
                // For actor (Arc<Mutex<T>>) variables, lock first.
                if let Some(first) = args.first() {
                    if let ExprKind::Var(v) = &first.value.kind {
                        if self.var_mutex_types.contains(v.as_str()) && self.in_async {
                            return format!("{}.len()", self.actor_read_access(v));
                        }
                    }
                }
                let a = self.emit_expr(&args[0].value);
                format!("{}.len()", a)
            }
            "int"   => format!("({} as i64)", self.emit_expr(&args[0].value)),
            "uint"  => format!("({} as u64)", self.emit_expr(&args[0].value)),
            "float" => format!("({} as f64)", self.emit_expr(&args[0].value)),
            "str"   => {
                // Single non-string arg → conversion.
                // String first arg (with optional extra args) → format like format().
                if let Some(first) = args.first() {
                    if matches!(&first.value.kind, ExprKind::StringInterp(_)) || args.len() >= 2 {
                        return self.emit_print_call_named("format", args);
                    }
                }
                format!("{}.to_string()", self.emit_expr(&args[0].value))
            }
            // Math functions: boring global → Rust method on f64.
            // Cast argument to f64 to avoid "ambiguous numeric type" errors on literals.
            "sqrt"       => format!("({} as f64).sqrt()", self.emit_expr(&args[0].value)),
            "abs"        => format!("({} as f64).abs()", self.emit_expr(&args[0].value)),
            "floor"      => format!("({} as f64).floor()", self.emit_expr(&args[0].value)),
            "ceil"       => format!("({} as f64).ceil()", self.emit_expr(&args[0].value)),
            "round"      => format!("({} as f64).round()", self.emit_expr(&args[0].value)),
            "sin"        => format!("({} as f64).sin()", self.emit_expr(&args[0].value)),
            "cos"        => format!("({} as f64).cos()", self.emit_expr(&args[0].value)),
            "tan"        => format!("({} as f64).tan()", self.emit_expr(&args[0].value)),
            "asin"       => format!("({} as f64).asin()", self.emit_expr(&args[0].value)),
            "acos"       => format!("({} as f64).acos()", self.emit_expr(&args[0].value)),
            "atan"       => format!("({} as f64).atan()", self.emit_expr(&args[0].value)),
            "atan2"      => {
                let y = self.emit_expr(&args[0].value);
                let x = self.emit_expr(&args[1].value);
                format!("({} as f64).atan2({} as f64)", y, x)
            }
            "exp"        => format!("({} as f64).exp()", self.emit_expr(&args[0].value)),
            "log"        => format!("({} as f64).ln()", self.emit_expr(&args[0].value)),
            "log2"       => format!("({} as f64).log2()", self.emit_expr(&args[0].value)),
            "log10"      => format!("({} as f64).log10()", self.emit_expr(&args[0].value)),
            "pow"        => {
                let b = self.emit_expr(&args[0].value);
                let e = self.emit_expr(&args[1].value);
                format!("({} as f64).powf({} as f64)", b, e)
            }
            "min"        => {
                if args.len() == 1 {
                    format!("{}.iter().cloned().reduce(f64::min).expect(\"cannot compute min of empty collection\")", self.emit_expr(&args[0].value))
                } else {
                    let a = self.emit_expr(&args[0].value);
                    let b = self.emit_expr(&args[1].value);
                    format!("({}).min({})", a, b)
                }
            }
            "max"        => {
                if args.len() == 1 {
                    format!("{}.iter().cloned().reduce(f64::max).expect(\"cannot compute max of empty collection\")", self.emit_expr(&args[0].value))
                } else {
                    let a = self.emit_expr(&args[0].value);
                    let b = self.emit_expr(&args[1].value);
                    format!("({}).max({})", a, b)
                }
            }
            "clamp"      => {
                let x  = self.emit_expr(&args[0].value);
                let lo = self.emit_expr(&args[1].value);
                let hi = self.emit_expr(&args[2].value);
                format!("({}).clamp({}, {})", x, lo, hi)
            }
            "sign"       => format!("({}).signum()", self.emit_expr(&args[0].value)),
            "isNaN"      => format!("({}).is_nan()", self.emit_expr(&args[0].value)),
            "isInfinite" => format!("({}).is_infinite()", self.emit_expr(&args[0].value)),
            "readLine"   => {
                // Emit Arc<str> so the result is directly usable as a `string` value.
                "{ let mut __line = String::new(); std::io::stdin().read_line(&mut __line).expect(\"failed to read from stdin\"); Arc::<str>::from(__line.trim()) }".into()
            }
            // drop(x) — explicitly releases ownership, maps directly to Rust's drop()
            "drop" => {
                let a = self.emit_expr(&args[0].value);
                format!("drop({})", a)
            }
            "args" => {
                "std::env::args().skip(1).map(|s| Arc::<str>::from(s)).collect::<Vec<_>>()".into()
            }
            "ord" => {
                let s = self.emit_expr(&args[0].value);
                format!("({}).chars().next().expect(\"ord: empty string\") as i64", s)
            }
            "chr" => {
                let n = self.emit_expr(&args[0].value);
                format!("Arc::<str>::from(char::from_u32({} as u32).expect(\"chr: invalid codepoint\").to_string())", n)
            }
            "exit" => {
                let code = self.emit_expr(&args[0].value);
                format!("{{ std::process::exit({} as i32) }}", code)
            }
            // json(v) → serde_json::to_string(&v).unwrap_or_default()
            "json" => {
                self.uses_serde.set(true);
                let a = self.emit_expr(&args[0].value);
                format!("serde_json::to_string(&{}).unwrap_or_default()", a)
            }
            _ => {
                // Look up registered signature for optional-arg coercion
                let args_s = self.emit_args_coerced(name, args);
                format!("{}({})", escape_rust_keyword(name), args_s)
            }
        }
    }

    pub(crate) fn emit_print_call(&self, newline: bool, args: &[Arg]) -> String {
        let macro_name = if newline { "println" } else { "print" };
        self.emit_print_call_named(macro_name, args)
    }

    pub(crate) fn emit_print_call_named(&self, macro_name: &str, args: &[Arg]) -> String {
        if args.is_empty() {
            return format!("{}!()", macro_name);
        }
        // Positional substitution: `print "..{}..", expr, expr2`
        // First arg is a string template where `{}` holes bind to extra args in order.
        // Inline `{name}` holes are interleaved naturally (left-to-right).
        if args.len() >= 2 {
            if let ExprKind::StringInterp(segs) = &args[0].value.kind {
                let positional: Vec<String> = args[1..].iter()
                    .map(|a| self.emit_expr(&a.value))
                    .collect();
                let (fmt, combined) = self.build_positional_format(segs, &positional);
                return if combined.is_empty() {
                    format!("{}!(\"{}\")", macro_name, fmt)
                } else {
                    format!("{}!(\"{}\", {})", macro_name, fmt, combined.join(", "))
                };
            }
        }
        // If the single arg is a string interp, unfold it
        if args.len() == 1 {
            if let ExprKind::StringInterp(segs) = &args[0].value.kind {
                let (fmt, extra_args) = self.build_format_string(segs);
                return if extra_args.is_empty() {
                    format!("{}!(\"{}\")", macro_name, fmt)
                } else {
                    format!("{}!(\"{}\", {})", macro_name, fmt, extra_args.join(", "))
                };
            }
            if let ExprKind::Str(s) = &args[0].value.kind {
                return format!("{}!(\"{}\")", macro_name, escape_str(s));
            }
        }
        // General case: println!("{}", arg) or println!("{} {}", a, b)
        // Vec collections use BoringFmt(&v) with "{}" so strings show without debug quotes.
        // HashMap/HashSet fall back to "{:?}" since they have no Display impl.
        // Optional values are unwrapped: `bm1` (Option<T>) → `bm1.as_ref().map_or(...)`.
        let args_with_specs: Vec<(String, &str)> = args.iter().map(|a| {
            let is_optional_var = matches!(&a.value.kind,
                ExprKind::Var(n) if self.optional_vars.contains(n.as_str()));
            // Also detect function calls that return Optional types.
            let is_optional_call = !is_optional_var && matches!(&a.value.kind,
                ExprKind::Call(callee, _) | ExprKind::GenericCall(callee, _, _)
                if matches!(&callee.kind, ExprKind::Var(n)
                    if matches!(self.fn_return_types.get(n.as_str()), Some(Type::Optional(_)))));
            let is_optional = is_optional_var || is_optional_call;
            let expr_s = if is_optional {
                let v = self.emit_expr(&a.value);
                format!("{}.as_ref().map_or_else(|| \"nil\".to_string(), |v| format!(\"{{}}\", v))", v)
            } else {
                self.emit_expr(&a.value)
            };
            if is_optional {
                return (expr_s, "{}");
            }
            let is_vec_var = matches!(&a.value.kind, ExprKind::Var(n) if self.vec_vars.contains(n.as_str()));
            let is_col = looks_like_collection(&expr_s)
                || matches!(&a.value.kind, ExprKind::Var(n) if self.collection_vars.contains(n.as_str()))
                || matches!(&a.value.kind, ExprKind::Array(_))
                || self.expr_returns_collection(&a.value);
            let (expr_s, spec) = boring_vec_fmt(expr_s, is_col, is_vec_var);
            (expr_s, spec)
        }).collect();
        let placeholders: String = args_with_specs.iter().map(|(_, s)| *s).collect::<Vec<_>>().join(" ");
        let args_s: Vec<String> = args_with_specs.into_iter().map(|(e, _)| e).collect();
        format!("{}!(\"{}\", {})", macro_name, placeholders, args_s.join(", "))
    }

}
