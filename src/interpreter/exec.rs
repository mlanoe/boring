use super::*;
use std::collections::HashMap;
use std::rc::Rc;

impl Interpreter {
    pub(crate) fn exec_stmt(&mut self, stmt: &Stmt, env: EnvRef) -> Result<(), Signal> {
        match stmt {
            Stmt::Let(s) => {
                // `static let/var` → store in global env, initialise only once
                let target_env: EnvRef = if s.is_static {
                    Rc::clone(&self.global)
                } else {
                    Rc::clone(&env)
                };
                // Skip re-initialisation for static vars already defined
                if s.is_static && target_env.borrow().get(&s.name).is_some() {
                    // Variable already initialised in a previous call; just make it
                    // visible in the local scope via an alias lookup (reads go through
                    // the env chain, writes need the name present locally for mutable
                    // statics).  We don't redefine — nothing to do here.
                    return Ok(());
                }
                if let Some(ty) = &s.ty {
                    self.check_type_has_qualifier(ty, s.line)?;
                }
                // `lazy` binding — always uninitialized, awaits first `?=`.
                if s.is_lazy || matches!(s.binding, BindingKind::Lazy) {
                    target_env.borrow_mut().define_lazy(&s.name);
                    return Ok(());
                }
                // Deferred initialisation (`let v` / `var v` without `= expr`).
                // Always define as mutable so the first assignment in a branch is allowed.
                // The transpiler emits `let v;` / `let mut v;` and Rust enforces single-assignment.
                if s.value.is_none() {
                    target_env.borrow_mut().define_mut(&s.name, Value::Uninitialized);
                    return Ok(());
                }
                let s_val = s.value.as_ref().unwrap();
                Self::check_no_owned_extract(s_val, &env, s.line)?;
                let val = self.eval_expr(s_val, Rc::clone(&env))?;
                // Apply type coercion if annotation is present (e.g. Int literal → Uint).
                // Also try implicit user-defined `as T:` conversion when the value doesn't
                // directly match the annotation (e.g. `let Animal a = dog`).
                let val = if let Some(ty) = &s.ty {
                    let resolved = self.resolve_type(ty);
                    let coerced = Self::coerce_to_type(val, &resolved);
                    if !self.value_matches_type(&coerced, &resolved) {
                        match self.cast_value(coerced.clone(), &resolved, s.line) {
                            Ok(converted) if self.value_matches_type(&converted, &resolved) => converted,
                            // Coercion and cast both failed — if we have a concrete type
                            // annotation (not a type param), raise a type error rather than
                            // silently binding the wrong-typed value.
                            _ => {
                                // Placeholder `Named("_")` is used for qualifier-on-name bindings
                                // like `let b'weak = a` — not a concrete type annotation.
                                let is_inferred = Self::is_inferred_type(&resolved);
                                let is_concrete = !is_inferred;
                                if is_concrete {
                                    return Err(err(
                                        format!(
                                            "cannot assign {} to '{}': expected {}",
                                            coerced.type_name(),
                                            s.name,
                                            Self::display_type(&resolved),
                                        ),
                                        s.line,
                                    ));
                                }
                                coerced
                            }
                        }
                    } else { coerced }
                } else { val };
                // Capture copy-ness before `val` is consumed by define()
                let val_is_copy = Self::is_copy_value(&val);
                let is_shared_ty = s.ty.as_ref().map(|ty| {
                    matches!(self.resolve_type(ty), Type::Qualified(_, OwnerQual::Shared))
                }).unwrap_or(false);
                let is_actor_ty = s.ty.as_ref().map(|ty| {
                    matches!(self.resolve_type(ty), Type::Qualified(_, OwnerQual::Actor | OwnerQual::ActorTask | OwnerQual::Guard | OwnerQual::GuardTask))
                }).unwrap_or(false);
                // Propagate shared/actor from source when no type annotation.
                let src_var_name = if let ExprKind::Var(src) = &s_val.kind { Some(src.clone()) } else { None };
                let src_is_shared = !is_shared_ty && !is_actor_ty && src_var_name.as_ref()
                    .map(|s| env.borrow().is_shared(s.as_str()))
                    .unwrap_or(false);
                let src_is_actor = !is_shared_ty && !is_actor_ty && src_var_name.as_ref()
                    .map(|s| env.borrow().is_actor(s.as_str()))
                    .unwrap_or(false);
                // `let` = immutable (no def methods, no reassign).
                // `var T'shared` = reassignable only (Arc<T> has no interior mutability,
                //   def methods are still forbidden — use T'actor for that).
                // `var` = fully mutable (reassign + def methods).
                let is_shared_var = s.binding.is_mutable() && (is_shared_ty || src_is_shared);
                if is_shared_var {
                    target_env.borrow_mut().define_shared_mut(&s.name, val);
                } else if s.binding.is_mutable() {
                    target_env.borrow_mut().define_mut(&s.name, val);
                } else {
                    target_env.borrow_mut().define(&s.name, val);
                }
                // Content-mutation permission (`def` calls, field writes,
                // collection mutation) is independent of the rebind axis above —
                // `var Point p` alone no longer grants it; see `binding_grants_mut`.
                if crate::ast::binding_grants_mut(&s.binding, s.var_mut, s.ty.as_ref()) {
                    target_env.borrow_mut().mark_content_mutable(&s.name);
                }
                if let Some(ty) = &s.ty {
                    target_env.borrow_mut().mark_declared_type(&s.name, ty.clone());
                    let resolved = self.resolve_type(ty);
                    // Owned-element collection: track + invalidate sources
                    if Self::type_has_owned_elems(&resolved) {
                        target_env.borrow_mut().mark_owned_collection(&s.name);
                        Self::invalidate_owned_collection_sources(&resolved, s_val, &target_env);
                    }
                    // Task-safe qualifier: mark var so task captures are allowed
                    if Self::type_annotation_is_task_safe(&resolved) {
                        target_env.borrow_mut().mark_task_safe(&s.name);
                    }
                    // Owned qualifier: mark var so task captures invalidate the source
                    if matches!(resolved, Type::Qualified(_, OwnerQual::Owned)) {
                        target_env.borrow_mut().mark_owned_var(&s.name);
                    }
                    // Track interior-mutable qualifiers (still gated by `mut`/`var` for `def`
                    // calls, like every other binding — this flag only exempts `'actor`/`'guard`
                    // from the separate `'shared` "no interior mutability" diagnostic below).
                    if is_actor_ty {
                        target_env.borrow_mut().mark_actor(&s.name);
                    }
                    // let T'shared: mark as shared (def methods forbidden, no move on assign).
                    if is_shared_ty {
                        target_env.borrow_mut().shared_bindings.insert(s.name.clone());
                    }
                }
                // Propagate shared/actor to dest when inferred from source.
                if src_is_shared {
                    target_env.borrow_mut().shared_bindings.insert(s.name.clone());
                }
                if src_is_actor {
                    target_env.borrow_mut().mark_actor(&s.name);
                }
                // Move by default: `let b = a` moves non-copy values.
                // Primitive and shared types (Int, Float, Bool, Str, Nil) are Copy and are not moved.
                // Borrow annotations (`T&`, `var T& ref = p`) alias rather than move.
                // 'shared/'actor/'guard bindings are reference-counted — assignment is an alias.
                let is_borrow = s.ty.as_ref().map(|ty| {
                    matches!(self.resolve_type(ty), Type::Qualified(_, OwnerQual::Borrow | OwnerQual::BorrowMut | OwnerQual::BorrowShared))
                }).unwrap_or(false);
                let src_is_rc_like = src_var_name.as_ref().map(|s| {
                    env.borrow().is_shared(s.as_str()) || env.borrow().is_actor(s.as_str())
                }).unwrap_or(false);
                if !val_is_copy && !is_borrow && !src_is_rc_like {
                    if let ExprKind::Var(src) = &s_val.kind {
                        env.borrow_mut().set_moved(src);
                    }
                }
                Ok(())
            }
            Stmt::LetDestructure(s) => {
                let val = self.eval_expr(&s.value, Rc::clone(&env))?;
                let elems = match val {
                    Value::Tuple(ref v) => v.clone(),
                    // Allow destructuring an Array as if it were a tuple
                    Value::Array(ref v) => v.as_ref().clone(),
                    other => {
                        return Err(err(
                            format!("cannot destructure '{}' as a tuple", other.type_name()),
                            s.line,
                        ));
                    }
                };
                for (i, binding) in s.bindings.iter().enumerate() {
                    if binding.name == "_" { continue; }
                    let mut v = elems.get(i).cloned().unwrap_or(Value::Nil);
                    if let Some(ty) = &binding.ty {
                        let resolved = self.resolve_type(ty);
                        v = Self::coerce_to_type(v, &resolved);
                    }
                    // Each slot's own resolved binding, not the statement's
                    // overall one — they can differ per-element
                    // (docs/book.md).
                    if binding.binding.is_mutable() {
                        env.borrow_mut().define_mut(&binding.name, v);
                    } else {
                        env.borrow_mut().define(&binding.name, v);
                    }
                    if crate::ast::binding_grants_mut(&binding.binding, binding.var_mut, binding.ty.as_ref()) {
                        env.borrow_mut().mark_content_mutable(&binding.name);
                    }
                    if let Some(ty) = &binding.ty {
                        env.borrow_mut().mark_declared_type(&binding.name, ty.clone());
                    }
                }
                Ok(())
            }
            Stmt::Return(s) => {
                let val = match &s.value {
                    Some(e) => {
                        Self::check_no_owned_extract(e, &env, s.line)?;
                        // `return .Left` — resolve against the enclosing function's own
                        // declared return type (return_ty_stack, pushed by call_fn) instead
                        // of eval_expr's ambiguous-scan fallback, mirroring emit_flow.rs's
                        // transpiler-side fix for the same case.
                        let hinted = match (&e.kind, self.return_ty_stack.last()) {
                            (ExprKind::DotIdent(name), Some(Some(ty))) => self.resolve_dot_ident_hint(name, ty),
                            _ => None,
                        };
                        match hinted {
                            Some(v) => v,
                            None => self.eval_expr(e, Rc::clone(&env))?,
                        }
                    }
                    None => Value::Nil,
                };
                Err(Signal::Return(val))
            }
            Stmt::Break(_, val_expr) => {
                let val = match val_expr {
                    Some(e) => self.eval_expr(e, Rc::clone(&env))?,
                    None    => Value::Void,
                };
                Err(Signal::Break(val))
            }
            Stmt::Continue(_) => Err(Signal::Continue),
            Stmt::Throw(s) => {
                let val = match &s.value {
                    Some(e) => self.eval_expr(e, Rc::clone(&env))?,
                    None => {
                        // re-throw: look up `error` in scope
                        env.borrow().get("error").unwrap_or(Value::Nil)
                    }
                };
                Err(Signal::Exception(val))
            }
            Stmt::If(s) => self.exec_if(s, env),
            Stmt::IfLet(s) => self.exec_if_let(s, env),
            Stmt::Match(s) => self.exec_match(s, env),
            Stmt::While(s) => self.exec_while(s, env),
            Stmt::WhileLet(s) => self.exec_while_let(s, env),
            Stmt::DoWhile(s) => self.exec_do_while(s, env),
            Stmt::Loop(s) => self.exec_loop(s, env),
            Stmt::Wait(_, _) => Ok(()), // no-op in synchronous interpreter
            Stmt::For(s) => self.exec_for(s, env),
            Stmt::Try(s) => self.exec_try(s, env),
            Stmt::Expr(e) => {
                // Mutating method calls on a variable auto-assign the result back.
                // The receiver expression is evaluated exactly ONCE here; calling
                // `eval_expr(e)` a second time would re-evaluate it (bug for
                // non-idempotent receivers like `getArray().push(x)`).
                const MUTATING: &[&str] = &["push", "append", "insert", "remove", "sort", "sortBy", "reverse", "add", "removeAt", "pop"];
                // "set"/"put" are mutating on Dict/Set but NOT on user-defined structs
                const MUTATING_COLL_ONLY: &[&str] = &["set", "put"];
                if let ExprKind::MethodCall(obj_expr, method, args) = &e.kind {
                    let might_mutate = MUTATING.contains(&method.as_str())
                        || MUTATING_COLL_ONLY.contains(&method.as_str());
                    if might_mutate {
                        // Fast path: `var_name.mutatingArrayMethod(args)` — this is the
                        // form loop bodies use (`samples.push(x)` as its own statement),
                        // so it's the hot path the O(n) full-array clone below needs to
                        // avoid. See `try_fast_mutating_array_call`'s doc comment.
                        let line = e.line;
                        if let ExprKind::Var(name) = &obj_expr.kind {
                            if let Some(result) = self.try_fast_mutating_array_call(name, method, args, &env, line) {
                                result?;
                                if matches!(method.as_str(), "push" | "append")
                                    && env.borrow().is_owned_collection(name)
                                {
                                    if let Some(n) = args.first().and_then(|arg| {
                                        if let ExprKind::Var(n) = &arg.value.kind { Some(n.clone()) } else { None }
                                    }) {
                                        env.borrow_mut().invalidate(&n);
                                    }
                                }
                                return Ok(());
                            }
                        }
                        // Evaluate the receiver once — never again.
                        let obj_val = self.eval_expr(obj_expr, Rc::clone(&env))?;
                        let is_collection = matches!(&obj_val, Value::Array(_) | Value::Dict(_) | Value::Set(_));
                        let is_coll_mutating = MUTATING_COLL_ONLY.contains(&method.as_str())
                            && matches!(&obj_val, Value::Dict(_) | Value::Set(_));

                        if (MUTATING.contains(&method.as_str()) && is_collection) || is_coll_mutating {
                            // Determine if we need to invalidate an owned arg after the call
                            let invalidate_name: Option<String> =
                                if matches!(method.as_str(), "push" | "append") {
                                    if let ExprKind::Var(coll_name) = &obj_expr.kind {
                                        if env.borrow().is_owned_collection(coll_name) {
                                            args.first().and_then(|arg| {
                                                if let ExprKind::Var(n) = &arg.value.kind {
                                                    Some(n.clone())
                                                } else { None }
                                            })
                                        } else { None }
                                    } else { None }
                                } else { None };

                            // Evaluate args, then call the method with the already-evaluated receiver
                            let arg_vals = self.eval_args(args, Rc::clone(&env))?;
                            let mut out_self: Option<Value> = None;
                            let result = self.call_method(obj_val, method, arg_vals, line, &mut out_self)?;

                            // Write back the mutated collection.
                            // Two cases:
                            //  (a) push/append/insert/sort/… → result IS the new collection
                            //  (b) pop/remove/removeAt → result is the extracted element;
                            //      call_method sets out_self to the shortened collection
                            if matches!(result, Value::Array(_) | Value::Dict(_) | Value::Set(_)) {
                                self.assign(obj_expr, result, Rc::clone(&env), line)?;
                            } else if let Some(new_coll) = out_self {
                                if matches!(new_coll, Value::Array(_) | Value::Dict(_) | Value::Set(_)) {
                                    self.assign(obj_expr, new_coll, Rc::clone(&env), line)?;
                                }
                            }

                            if let Some(name) = invalidate_name {
                                env.borrow_mut().invalidate(&name);
                            }
                            return Ok(());
                        }
                        // Method name matched but receiver is not a collection
                        // (e.g. a struct with a user-defined `push` method).
                        // Fall through — the struct's call_method / out_self write-back
                        // path handles write-back correctly.
                    }
                }
                let val = self.eval_expr(e, Rc::clone(&env))?;
                // Must-use: bare call whose return value is silently discarded is an error.
                // Only enforced for Call / MethodCall expressions (not operators, blocks, etc.).
                // Void functions return Value::Void — those are always OK as bare statements.
                let is_bare_call = matches!(&e.kind,
                    ExprKind::Call(..) | ExprKind::MethodCall(..) | ExprKind::GenericCall(..)
                );
                // In a `kernel:` context, a bare `k(block=N)` call returns a KernelHandle.
                // The `exec_kernel_block` handler writes it back — suppress the must-use error.
                let is_kernel_handle = matches!(val, Value::KernelHandle { .. });
                if is_bare_call && !matches!(val, Value::Void | Value::Nil) && !(self.kernel_context && is_kernel_handle) {
                    return Err(err(
                        "return value discarded — bind it with `let`, discard with `_ = f()`",
                        e.line,
                    ));
                }
                Ok(())
            }
            Stmt::Defer(stmts) => {
                // Register this block in the current function's defer frame
                if let Some(frame) = self.defer_stack.last_mut() {
                    frame.push(stmts.clone());
                }
                Ok(())
            }
            Stmt::Guard(g) => self.exec_guard(g, env),
            Stmt::Fn(decl) => {
                let val = Value::Fn { decl: decl.clone(), captured: Rc::clone(&env) };
                env.borrow_mut().define(&decl.name, val);
                Ok(())
            }
            Stmt::Struct(decl) => {
                // Struct declared inside a function body — register in local scope only,
                // NOT in global, so it is invisible outside the function.
                self.exec_item(&Item::Struct(decl.clone()), Rc::clone(&env))?;
                // Undo the global registration that exec_item performed.
                if self.global.borrow().vars.contains_key(&decl.name) {
                    // Only remove it if the global wasn't defining this type before.
                    // We detect "local" by checking whether env IS the global.
                    if !Rc::ptr_eq(&env, &self.global) {
                        self.global.borrow_mut().vars.remove(&decl.name);
                    }
                }
                Ok(())
            }
            Stmt::Enum(decl) => {
                self.exec_item(&Item::Enum(decl.clone()), env)
            }
            Stmt::Alias(decl) => {
                // Local type alias — added to the global alias map like top-level aliases.
                // Type aliases are resolved at type-check time, so scoping is not needed.
                self.aliases.insert(decl.name.clone(), decl.ty.clone());
                Ok(())
            }
            Stmt::Mod(decl) => {
                for item in &decl.items {
                    self.exec_item(item, Rc::clone(&env))?;
                }
                Ok(())
            }
            Stmt::Yield(expr, _line) => {
                let val = self.eval_expr(expr, Rc::clone(&env))?;
                // Inside a stream body (in_stream=true) yields are collected as side effects
                // so that `for` loops and other control flow continue normally.
                self.stream_yields.push(val);
                Ok(())
            }
            // `sync` parses as `Stmt::Comment("sync")` (see parse_stmt.rs) — a real
            // block-level barrier when the kernel has `'sync` fields (`kernel_barrier`
            // is only set on the per-thread interpreters `run_kernel_parallel` builds
            // for such a kernel); an ordinary comment (no-op) otherwise.
            Stmt::Comment(c) if c == "sync" => {
                if let Some(barrier) = &self.kernel_barrier {
                    barrier.wait();
                }
                Ok(())
            }
            Stmt::Comment(_) => Ok(()),
            Stmt::KernelBlock(s) => {
                self.exec_kernel_block(&s.body, env)
            }
            // `with <name>:` — a no-op wrapper under the interpreter. `GpuQual` (the
            // kernel-field qualifier enum) is never referenced anywhere in this module:
            // a single-threaded tree-walk simulation has no host/device split to model,
            // so every kernel-context qualifier already behaves as a plain value here.
            // The same precedent extends to the host-context qualifiers this statement
            // exists for (`'gpu'unified`/`'gpu'global` residency, `'actor`/`'guard`
            // per-block locking) — there is nothing to acquire or write back; the body
            // just runs directly. See docs/scoped-access-blocks.md, "Cross-target behavior".
            Stmt::With(s) => {
                let child = Env::child(Rc::clone(&env));
                self.exec_block(&s.body, child)
            }
        }
    }

    /// Evaluate a list of CondClauses into a child environment.
    /// Returns `Some(child_env)` if ALL clauses pass, `None` if any fails.
    /// `Let` bindings are defined in `child_env` as they are evaluated.
    pub(crate) fn eval_cond_clauses(
        &mut self,
        clauses: &[CondClause],
        child: &EnvRef,
    ) -> Result<bool, Signal> {
        for clause in clauses {
            match clause {
                CondClause::Let(name, expr) => {
                    let val = self.eval_expr(expr, Rc::clone(child))?;
                    match val {
                        Value::Nil => return Ok(false),
                        other => child.borrow_mut().define(name, other),
                    }
                }
                CondClause::LetPat(pat, expr) => {
                    let val = self.eval_expr(expr, Rc::clone(child))?;
                    let mut bindings = std::collections::HashMap::new();
                    let mut mut_names = std::collections::HashSet::new();
                    if !self.match_pattern(pat, &val, &mut bindings, &mut mut_names) {
                        return Ok(false);
                    }
                    for (k, v) in bindings {
                        child.borrow_mut().define(&k, v);
                        if mut_names.contains(&k) {
                            child.borrow_mut().mark_content_mutable(&k);
                        }
                    }
                }
                CondClause::Expr(expr) => {
                    let val = self.eval_expr(expr, Rc::clone(child))?;
                    let b = self.expect_bool(val, expr.line)?;
                    if !b { return Ok(false); }
                }
            }
        }
        Ok(true)
    }

    pub(crate) fn exec_if_let(&mut self, s: &IfLetStmt, env: EnvRef) -> Result<(), Signal> {
        let child = Env::child(Rc::clone(&env));
        if self.eval_cond_clauses(&s.clauses, &child)? {
            return self.exec_block(&s.then_body, child);
        }
        for branch in &s.elif_branches {
            let elif_child = Env::child(Rc::clone(&env));
            if self.eval_cond_clauses(&branch.clauses, &elif_child)? {
                return self.exec_block(&branch.body, elif_child);
            }
        }
        if let Some(else_body) = &s.else_body {
            let else_child = Env::child(Rc::clone(&env));
            self.exec_block(else_body, else_child)?;
        }
        Ok(())
    }

    pub(crate) fn exec_guard(&mut self, g: &GuardStmt, env: EnvRef) -> Result<(), Signal> {
        match &g.cond {
            GuardCond::Expr(e) => {
                let val = self.eval_expr(e, Rc::clone(&env))?;
                let b = self.expect_bool(val, e.line)?;
                if !b {
                    self.exec_block(&g.else_body, Rc::clone(&env))?;
                }
                Ok(())
            }
            GuardCond::Clauses(clauses) => {
                // Guard bindings go into the CURRENT scope (visible after the guard)
                let passed = self.eval_cond_clauses(clauses, &env)?;
                if !passed {
                    self.exec_block(&g.else_body, Rc::clone(&env))?;
                }
                Ok(())
            }
        }
    }

    pub(crate) fn exec_block(&mut self, stmts: &[Stmt], env: EnvRef) -> Result<(), Signal> {
        for stmt in stmts {
            self.exec_stmt(stmt, Rc::clone(&env))?;
        }
        Ok(())
    }

    pub(crate) fn exec_if(&mut self, s: &IfStmt, env: EnvRef) -> Result<(), Signal> {
        for (cond, body) in &s.branches {
            let val = self.eval_expr(cond, Rc::clone(&env))?;
            let b = self.expect_bool(val, cond.line)?;
            if b {
                let child = Env::child(Rc::clone(&env));
                return self.exec_block(body, child);
            }
        }
        if let Some(else_body) = &s.else_body {
            let child = Env::child(Rc::clone(&env));
            self.exec_block(else_body, child)?;
        }
        Ok(())
    }

    pub(crate) fn exec_match(&mut self, s: &MatchStmt, env: EnvRef) -> Result<(), Signal> {
        let subject = self.eval_expr(&s.subject, Rc::clone(&env))?;
        'arms: for arm in &s.arms {
            for pattern in &arm.patterns {
                let mut bindings = HashMap::new();
                let mut mut_names = std::collections::HashSet::new();
                if self.match_pattern(pattern, &subject, &mut bindings, &mut mut_names) {
                    let child = Env::child(Rc::clone(&env));
                    for (k, v) in bindings {
                        child.borrow_mut().define(&k, v);
                        if mut_names.contains(&k) {
                            child.borrow_mut().mark_content_mutable(&k);
                        }
                    }
                    // Evaluate optional guard in the child env (bindings already in scope)
                    if let Some(guard_expr) = &arm.guard {
                        let guard_val = self.eval_expr(guard_expr, Rc::clone(&child))?;
                        if !self.expect_bool(guard_val, guard_expr.line)? {
                            continue 'arms;
                        }
                    }
                    match &arm.body {
                        MatchBody::Expr(e) => { self.eval_expr(e, child)?; }
                        MatchBody::Block(stmts) => self.exec_block(stmts, child)?,
                    }
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    pub(crate) fn exec_while(&mut self, s: &WhileStmt, env: EnvRef) -> Result<(), Signal> {
        loop {
            let cond = self.eval_expr(&s.condition, Rc::clone(&env))?;
            let b = self.expect_bool(cond, s.condition.line)?;
            if !b { break; }
            let child = Env::child(Rc::clone(&env));
            match self.exec_block(&s.body, child) {
                Ok(()) => {}
                Err(Signal::Break(_)) => break,
                Err(Signal::Continue) => continue,
                Err(other) => return Err(other),
            }
        }
        Ok(())
    }

    pub(crate) fn exec_while_let(&mut self, s: &WhileLetStmt, env: EnvRef) -> Result<(), Signal> {
        loop {
            let val = self.eval_expr(&s.value, Rc::clone(&env))?;
            let child = Env::child(Rc::clone(&env));
            if let Some(pat) = &s.pattern {
                // `while let Some(x) = expr:` — pattern form
                let mut bindings = std::collections::HashMap::new();
                let mut mut_names = std::collections::HashSet::new();
                if !self.match_pattern(pat, &val, &mut bindings, &mut mut_names) { break; }
                for (k, v) in bindings {
                    child.borrow_mut().define(&k, v);
                    if mut_names.contains(&k) {
                        child.borrow_mut().mark_content_mutable(&k);
                    }
                }
            } else {
                // `while let name = expr:` — simple nil-check binding
                if matches!(val, Value::Nil) { break; }
                child.borrow_mut().define(&s.name, val);
            }
            match self.exec_block(&s.body, child) {
                Ok(()) => {}
                Err(Signal::Break(_)) => break,
                Err(Signal::Continue) => continue,
                Err(other) => return Err(other),
            }
        }
        Ok(())
    }

    pub(crate) fn exec_do_while(&mut self, s: &DoWhileStmt, env: EnvRef) -> Result<(), Signal> {
        loop {
            let child = Env::child(Rc::clone(&env));
            match self.exec_block(&s.body, child) {
                Ok(()) => {}
                Err(Signal::Break(_)) => break,
                Err(Signal::Continue) => {}
                Err(other) => return Err(other),
            }
            let cond = self.eval_expr(&s.condition, Rc::clone(&env))?;
            let b = self.expect_bool(cond, s.condition.line)?;
            if !b { break; }
        }
        Ok(())
    }

    pub(crate) fn exec_loop(&mut self, s: &LoopStmt, env: EnvRef) -> Result<(), Signal> {
        loop {
            let child = Env::child(Rc::clone(&env));
            match self.exec_block(&s.body, child) {
                Ok(()) => {}
                Err(Signal::Break(_)) => break,
                Err(Signal::Continue) => continue,
                Err(other) => return Err(other),
            }
        }
        Ok(())
    }

    pub(crate) fn eval_loop(&mut self, s: &LoopStmt, env: EnvRef) -> Eval {
        loop {
            let child = Env::child(Rc::clone(&env));
            match self.exec_block(&s.body, child) {
                Ok(()) => {}
                Err(Signal::Break(val)) => return Ok(val),
                Err(Signal::Continue) => continue,
                Err(other) => return Err(other),
            }
        }
    }

    pub(crate) fn exec_for(&mut self, s: &ForStmt, env: EnvRef) -> Result<(), Signal> {
        let iterable = self.eval_expr(&s.iterable, Rc::clone(&env))?;
        // Iterator protocol: Object with a `next()` method — call it in a loop.
        if let Value::Object(_) = &iterable {
            return self.exec_for_iterator(iterable, s, env);
        }
        let items = self.collect_iterable(iterable, s.iterable.line)?;
        for (idx, item) in items.into_iter().enumerate() {
            let child = Env::child(Rc::clone(&env));
            if s.vars.len() == 1 {
                child.borrow_mut().define(&s.vars[0], item);
            } else {
                match item {
                    // Tuple (from dict, .enumerate(), .zip(), etc.) → destructure
                    Value::Tuple(elems) => {
                        for (i, var) in s.vars.iter().enumerate() {
                            let val = elems.get(i).cloned().unwrap_or(Value::Nil);
                            child.borrow_mut().define(var, val);
                        }
                    }
                    // Scalar value + multiple vars → auto-enumerate:
                    //   `for i, v in arr:`  ≡  `for i, v in arr.enumerate():`
                    //   vars[0] = index (int), vars[1] = element, rest = Nil
                    other => {
                        child.borrow_mut().define(&s.vars[0], Value::Int(idx as i64));
                        if s.vars.len() >= 2 {
                            child.borrow_mut().define(&s.vars[1], other);
                        }
                        for var in s.vars.iter().skip(2) {
                            child.borrow_mut().define(var, Value::Nil);
                        }
                    }
                }
            }
            match self.exec_block(&s.body, child) {
                Ok(()) => {}
                Err(Signal::Break(_)) => break,
                Err(Signal::Continue) => continue,
                Err(other) => return Err(other),
            }
        }
        Ok(())
    }

    /// Iterator protocol for `for x in obj:` when `obj` is a user-defined struct
    /// with a `def T? next():` method.  Calls `next()` until it returns `nil`.
    pub(crate) fn exec_for_iterator(&mut self, mut iter_val: Value, s: &ForStmt, env: EnvRef) -> Result<(), Signal> {
        let line = s.iterable.line;
        loop {
            let mut out_self: Option<Value> = None;
            let item = self.call_method(iter_val.clone(), "next", vec![], line, &mut out_self)?;
            // `next()` mutates self — update our local copy of the iterator.
            if let Some(new_self) = out_self {
                iter_val = new_self;
            }
            // `nil` (None) signals exhaustion.
            if matches!(item, Value::Nil) { break; }
            let child = Env::child(Rc::clone(&env));
            if s.vars.len() == 1 {
                child.borrow_mut().define(&s.vars[0], item);
            } else if s.vars.len() >= 2 {
                match item {
                    Value::Tuple(elems) => {
                        for (i, var) in s.vars.iter().enumerate() {
                            let val = elems.get(i).cloned().unwrap_or(Value::Nil);
                            child.borrow_mut().define(var, val);
                        }
                    }
                    other => {
                        child.borrow_mut().define(&s.vars[0], Value::Nil);
                        child.borrow_mut().define(&s.vars[1], other);
                    }
                }
            }
            match self.exec_block(&s.body, child) {
                Ok(()) => {}
                Err(Signal::Break(_)) => break,
                Err(Signal::Continue) => continue,
                Err(other) => return Err(other),
            }
        }
        Ok(())
    }

    pub(crate) fn collect_iterable(&mut self, val: Value, line: usize) -> Result<Vec<Value>, Signal> {
        match val {
            Value::Array(elems) => Ok(Value::rc_vec_into_owned(elems)),
            Value::Set(elems) => Ok(elems),
            Value::Tuple(elems) => Ok(elems),
            Value::Dict(pairs) => Ok(pairs.into_iter().map(|(k, v)| Value::Tuple(vec![k, v])).collect()),
            Value::Range { start, end, inclusive } => {
                let iter: Box<dyn Iterator<Item = i64>> = if inclusive {
                    Box::new(start..=end)
                } else {
                    Box::new(start..end)
                };
                Ok(iter.map(Value::Int).collect())
            }
            Value::Str(s) => Ok(s.chars().map(|c| Value::Str(c.to_string())).collect()),
            Value::Channel { buf, is_sender, .. } => {
                if is_sender {
                    return Err(err("cannot iterate over a channel sender", line));
                }
                Ok(buf.borrow_mut().drain(..).collect())
            }
            other => Err(err(format!("'{}' is not iterable", other.type_name()), line)),
        }
    }

    pub(crate) fn exec_try(&mut self, s: &TryStmt, env: EnvRef) -> Result<(), Signal> {
        let child = Env::child(Rc::clone(&env));
        let result = self.exec_block(&s.body, child);

        let outcome = match result {
            Ok(()) => None,
            Err(Signal::Exception(val)) => Some(val),
            Err(other) => return Err(other),
        };

        if let Some(exc_val) = outcome {
            let exc_type = exc_val.type_name().to_string();
            // Actual variant name, when the thrown value is an enum variant — used
            // below to disambiguate `catch EnumName.VariantA:` from a sibling
            // `catch EnumName.VariantB:` clause on the same enum.
            let exc_variant = match &exc_val {
                Value::EnumVariant { variant, .. } => Some(variant.as_str()),
                _ => None,
            };
            let mut handled = false;
            for clause in &s.catch_clauses {
                let type_matches = clause.types.is_empty()
                    || clause.types.iter().any(|t| {
                        t == &exc_type
                            // `Float64` is accepted as a spelling of `Float` here —
                            // `Value::Float64::type_name()` returns "Float" (kept for
                            // backward compatibility with existing `catch Float:`
                            // clauses), but a `catch Float64:` clause is just as valid
                            // a way to name the same type (docs/float-width-types.md
                            // §2 — float/Float/float64/Float64 are all Type::Float64).
                            || (exc_type == "Float" && t == "Float64")
                    });
                // A clause naming a specific variant (`catch EnumName.Variant:`) only
                // matches when the thrown value actually carries that variant —
                // previously this was ignored entirely, so with several such clauses
                // on the same enum the first one always matched regardless of which
                // variant was thrown.
                let matches = type_matches
                    && clause
                        .variant
                        .as_deref()
                        .is_none_or(|v| exc_variant == Some(v));
                if matches {
                    let cenv = Env::child(Rc::clone(&env));
                    cenv.borrow_mut().define("error", exc_val.clone());
                    self.exec_block(&clause.body.clone(), cenv)?;
                    handled = true;
                    break;
                }
            }
            if !handled {
                return Err(Signal::Exception(exc_val));
            }
        }

        Ok(())
    }

    // ─── Type alias resolution ───────────────────────────────────────────────

    /// Expand a `Type::Named` through the alias table (one level).
    /// `Type::Named("int")` → `Type::Qualified(Type::Int, Copy)` via built-in alias.
    pub(crate) fn resolve_type(&self, ty: &Type) -> Type {
        match ty {
            Type::Named(name) => {
                // Check type_param_stack first — a Named type might be a type param
                // (e.g. multi-letter type params like <Elem> are stored as Named in field types)
                for frame in self.type_param_stack.iter().rev() {
                    if let Some(bound) = frame.get(name.as_str()) {
                        return self.resolve_type(bound);
                    }
                }
                // Then check aliases — recurse (like the type_param_stack branch just
                // above) so a chained alias (`use Score as int`, where `int` is itself
                // a builtin alias for `Type::Int` — see `register_misc_globals`) fully
                // resolves to the primitive, not just one level to `Named("int")`. That
                // half-resolved `Named("int")` doesn't match `Value::Int` in
                // `value_matches_type` (which only recognizes the real `Type::Int`
                // variant, not an arbitrary `Named` string), so `let Score top = 100`
                // failed with a bogus "cannot assign Int to 'top': expected int" even
                // though the value and the fully-resolved type agree exactly.
                if let Some(expanded) = self.aliases.get(name) {
                    self.resolve_type(expanded)
                } else {
                    ty.clone()
                }
            }
            Type::TypeParam(name) => {
                // Look up in type_param_stack (single-letter params like T, K, V)
                for frame in self.type_param_stack.iter().rev() {
                    if let Some(bound) = frame.get(name.as_str()) {
                        return self.resolve_type(bound);
                    }
                }
                ty.clone()
            }
            Type::Optional(inner) => Type::Optional(Box::new(self.resolve_type(inner))),
            Type::Array(elem) => Type::Array(Box::new(self.resolve_type(elem))),
            Type::ArrayN(elem, n) => Type::ArrayN(Box::new(self.resolve_type(elem)), *n),
            // Resolve the element type (e.g. `Named("float")` -> its alias)
            // the same way Array/ArrayN do — axes carry no type to resolve.
            Type::LabeledArray(elem, axes) => Type::LabeledArray(Box::new(self.resolve_type(elem)), axes.clone()),
            Type::Set(elem) => Type::Set(Box::new(self.resolve_type(elem))),
            Type::Dict(k, v) => Type::Dict(
                Box::new(self.resolve_type(k)),
                Box::new(self.resolve_type(v)),
            ),
            Type::Tuple(elems) => Type::Tuple(elems.iter().map(|e| self.resolve_type(e)).collect()),
            Type::Qualified(inner, qual) => Type::Qualified(
                Box::new(self.resolve_type(inner)),
                qual.clone(),
            ),
            Type::Generic(name, args) => match name.as_str() {
                // Built-in collection generics: fold back into primitive collection types
                // so that type checking, qualifier checks, etc. all work uniformly.
                "Array" if args.len() == 1 =>
                    Type::Array(Box::new(self.resolve_type(&args[0]))),
                "Set" if args.len() == 1 =>
                    Type::Set(Box::new(self.resolve_type(&args[0]))),
                "Dict" if args.len() == 2 =>
                    Type::Dict(Box::new(self.resolve_type(&args[0])),
                               Box::new(self.resolve_type(&args[1]))),
                _ => Type::Generic(
                    name.clone(),
                    args.iter().map(|a| self.resolve_type(a)).collect(),
                ),
            },
            // `mut Type` — recurse so an aliased inner type (`mut int x`, parsed as
            // `Mut(Named("int"))` until alias resolution) actually resolves; the
            // previous default (`other => other.clone()`) left the inner `Named`
            // unresolved, which then failed `value_matches_type`'s `Named` branch
            // (looks for `Channel`/`Object`/`EnumVariant`, not a bare primitive).
            Type::Mut(inner) => Type::Mut(Box::new(self.resolve_type(inner))),
            // `LinkedList.Index` — resolve by looking up the struct's assoc type def.
            // Falls back to the unresolved AssocOf when the struct isn't found.
            Type::AssocOf(base, assoc) => {
                let type_name = match base.as_ref() {
                    Type::Named(n)      => n.clone(),
                    Type::Generic(n, _) => n.clone(),
                    _ => return ty.clone(),
                };
                // Structs are stored as `Value::Struct { decl, .. }` in the global env.
                if let Some(Value::Struct { decl, .. }) = self.global.borrow().get(&type_name) {
                    if let Some(def) = decl.assoc_type_defs.iter().find(|d| d.name == *assoc) {
                        return self.resolve_type(&def.ty.clone());
                    }
                }
                ty.clone()
            }
            other => other.clone(),
        }
    }

    // ─── Type coercion ───────────────────────────────────────────────────────

    /// Returns true for types that represent an inferred/placeholder annotation rather than
    /// a concrete type specified by the programmer. Used to decide whether a type mismatch
    /// should be silently accepted (inferred) or rejected with an error (concrete).
    ///
    /// Inferred types arise from qualifier-on-name syntax: `let b'weak = a` generates
    /// `ty = Qualified(Named("_"), Weak)` where `"_"` is a placeholder filled at runtime.
    pub(crate) fn is_inferred_type(ty: &Type) -> bool {
        match ty {
            Type::TypeParam(_) => true,
            Type::Dyn(_) | Type::Impl(_) => true,
            Type::Named(s) if s == "_" => true,
            Type::Qualified(inner, _) => Self::is_inferred_type(inner),
            Type::Optional(inner) => Self::is_inferred_type(inner),
            Type::LabeledArray(inner, _) => Self::is_inferred_type(inner),
            _ => false,
        }
    }

    /// Coerce a value to match a resolved type annotation when necessary.
    /// Currently handles Int → Uint coercion when the annotation is Uint (or Uint'copy).
    ///
    /// Negative integers are intentionally NOT coerced: `Int(-1)` remains `Int(-1)` so
    /// that the subsequent `value_matches_type` check fails and the caller emits a proper
    /// type error rather than silently wrapping -1 to 18446744073709551615.
    ///
    /// Recurses elementwise into `Value::Array`/`Value::Set` when the declared type is an
    /// array/set of a fixed-width numeric type (`Type::Array`/`ArrayN`/`LabeledArray`/`Set`)
    /// — an array-comprehension literal like `[0.0 for ..N]` evaluates its elements with no
    /// knowledge of the binding's declared element type, so it always produces untyped
    /// `Int`/`Float64` elements (`eval_expr` has no target-type hint to thread through a
    /// comprehension). Without this, `let [float32]'gpu'unified x = [0.0 for ..N]` left every
    /// element a `Float64`, which `value_matches_type`'s per-element check then rejected
    /// outright (`cannot assign Array to 'x': expected [float32]'gpu'unified`) since neither
    /// this function's scalar-only `base()` dispatch below nor `cast_value` (also scalar-only)
    /// ever looked inside the array to narrow its elements — confirmed via examples/saxpy.br.
    pub(crate) fn coerce_to_type(val: Value, ty: &Type) -> Value {
        fn strip_wrapper(ty: &Type) -> &Type {
            match ty {
                Type::Qualified(inner, _) | Type::Mut(inner) => strip_wrapper(inner),
                _ => ty,
            }
        }
        match (strip_wrapper(ty), &val) {
            (Type::Array(elem_ty), Value::Array(elems))
                | (Type::ArrayN(elem_ty, _), Value::Array(elems))
                | (Type::LabeledArray(elem_ty, _), Value::Array(elems)) => {
                return Value::Array(std::rc::Rc::new(
                    elems.iter().cloned().map(|e| Self::coerce_to_type(e, elem_ty)).collect(),
                ));
            }
            (Type::Set(elem_ty), Value::Set(elems)) => {
                return Value::Set(elems.iter().cloned().map(|e| Self::coerce_to_type(e, elem_ty)).collect());
            }
            _ => {}
        }

        fn base(ty: &Type) -> Option<Type> {
            match ty {
                Type::Uint | Type::Uint8
                    | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                    | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128
                    | Type::Float32 | Type::Float64 => Some(ty.clone()),
                Type::Qualified(inner, _) => base(inner),
                _ => None,
            }
        }
        match base(ty) {
            Some(Type::Uint) => match val {
                Value::Int(n) if n >= 0 => Value::Uint(n as u64),
                other => other, // negative Int: leave unchanged → type-check will reject it
            },
            Some(Type::Uint8) => match val {
                Value::Int(n) if (0..=255).contains(&n) => Value::Uint8(n as u8),
                other => other, // out-of-range Int: leave unchanged → type-check will reject it
            },
            Some(Type::Int8) => match val {
                Value::Int(n) if (i8::MIN as i64..=i8::MAX as i64).contains(&n) => Value::Int8(n as i8),
                other => other,
            },
            Some(Type::Int16) => match val {
                Value::Int(n) if (i16::MIN as i64..=i16::MAX as i64).contains(&n) => Value::Int16(n as i16),
                other => other,
            },
            Some(Type::Int32) => match val {
                Value::Int(n) if (i32::MIN as i64..=i32::MAX as i64).contains(&n) => Value::Int32(n as i32),
                other => other,
            },
            Some(Type::Int64) => match val {
                Value::Int(n) => Value::Int64(n),
                other => other,
            },
            Some(Type::Int128) => match val {
                Value::Int(n) => Value::Int128(n as i128),
                other => other,
            },
            Some(Type::Uint16) => match val {
                Value::Int(n) if (0..=u16::MAX as i64).contains(&n) => Value::Uint16(n as u16),
                other => other,
            },
            Some(Type::Uint32) => match val {
                Value::Int(n) if (0..=u32::MAX as i64).contains(&n) => Value::Uint32(n as u32),
                other => other,
            },
            Some(Type::Uint64) => match val {
                Value::Int(n) if n >= 0 => Value::Uint64(n as u64),
                other => other,
            },
            Some(Type::Uint128) => match val {
                Value::Int(n) if n >= 0 => Value::Uint128(n as u128),
                other => other,
            },
            // A bare `int` literal (e.g. `let float32 x = 1`) or an untyped `float`
            // literal (which evaluates to Value::Float64 by default, since float has
            // no independent flexible kind of its own — docs/float-width-types.md §2)
            // narrows to match the declared width at the binding site. This is
            // deliberately narrower in scope than full strict-mixing enforcement:
            // it coerces at declaration time (`let float32 x = float64_expr`), while
            // two independently-typed variables mixing in an *expression*
            // (`float32_var + float64_var`) is rejected by `eval_binop`'s stricter
            // check, which does have the source-expression context to distinguish
            // a literal from a resolved value and this function does not.
            Some(Type::Float32) => match val {
                Value::Int(n) => Value::Float32(n as f32),
                Value::Float64(f) => Value::Float32(f as f32),
                other => other,
            },
            Some(Type::Float64) => match val {
                Value::Int(n) => Value::Float64(n as f64),
                Value::Float32(f) => Value::Float64(f as f64),
                other => other,
            },
            _ => val,
        }
    }

    // ─── Type utilities ──────────────────────────────────────────────────────

    /// Extracts the base struct / enum name from a type, stripping any ownership qualifier.
    /// Returns `Some("Dog")` for `Dog'inline`, `Dog'auto`, `Dog`, etc.
    /// Used to resolve type aliases to their underlying constructors.
    pub(crate) fn type_base_name(ty: &Type) -> Option<String> {
        match ty {
            Type::Named(n) => Some(n.clone()),
            Type::Qualified(inner, _) => Self::type_base_name(inner),
            _ => None,
        }
    }

    /// Resolve `.Variant` against a known expected type — a function/method parameter's
    /// declared type, or the enclosing function's declared return type — instead of
    /// `eval_expr`'s `ExprKind::DotIdent` arm, which scans every enum registered globally
    /// and errors out as "ambiguous" the moment two enums share a variant name. This gives
    /// `boring run` the same param-type/return-type disambiguation the transpiler already
    /// does via `emit_let_value` (see `struct_method_param_types` in emit_methods.rs).
    /// Returns `None` when `hint_ty` doesn't name a known enum with this variant — callers
    /// fall back to the plain `eval_expr` path in that case, which still resolves the
    /// unambiguous case and errors on a genuine unresolved collision.
    pub(crate) fn resolve_dot_ident_hint(&self, name: &str, hint_ty: &Type) -> Option<Value> {
        let resolved = self.resolve_type(hint_ty);
        let base = Self::type_base_name(&resolved)?;
        if !self.enums.contains_key(&base) { return None; }
        let global = self.global.borrow();
        match global.get(&base) {
            Some(Value::EnumNamespace { variants, .. }) => variants.get(name).cloned(),
            _ => None,
        }
    }

    /// Build a per-position `.Variant` hint (see `resolve_dot_ident_hint`) for a call whose
    /// callee resolves to several same-named candidate `FnDecl`s (an overloaded free
    /// function or struct method). The right candidate is normally picked by
    /// `find_best_method`/`Value::OverloadedFn`'s dispatch in `call_value` — but that
    /// selection needs the *evaluated* argument `Value`s, which don't exist yet while we're
    /// still evaluating them.
    ///
    /// Position `i` gets a hint only when the arity-compatible candidates agree on a single
    /// enum there. "Agree" means: among candidates whose declared type at `i` is a *known
    /// enum*, there is exactly one such enum. A candidate with a non-enum type there (an
    /// `int` overload alongside a `TrafficLight` one, say) is simply irrelevant — a
    /// `.Variant` argument could never resolve against it anyway, so it doesn't block the
    /// hint. Two *different* enums both appearing at `i` is genuine ambiguity — picking one
    /// would silently guess which overload gets called — so that still yields no hint,
    /// falling back to the ambiguous-scan default exactly as before this method existed.
    pub(crate) fn merged_overload_param_hints(&self, candidates: &[FnDecl], arg_count: usize) -> Vec<Option<Type>> {
        let applicable: Vec<&FnDecl> = candidates.iter()
            .filter(|d| {
                let min_a = d.params.iter().filter(|p| p.default.is_none()).count();
                let max_a = d.params.len();
                arg_count >= min_a && arg_count <= max_a
            })
            .collect();
        let mut hints: Vec<Option<Type>> = vec![None; arg_count];
        if applicable.is_empty() { return hints; }
        for (i, hint) in hints.iter_mut().enumerate() {
            let mut distinct_enums: Vec<(String, Type)> = Vec::new();
            for d in &applicable {
                let Some(ty) = d.params.get(i).and_then(|p| p.ty.as_ref()) else { continue };
                let Some(base) = Self::type_base_name(&self.resolve_type(ty)) else { continue };
                if !self.enums.contains_key(&base) { continue; }
                if !distinct_enums.iter().any(|(b, _)| *b == base) {
                    distinct_enums.push((base, ty.clone()));
                }
            }
            if distinct_enums.len() == 1 {
                *hint = Some(distinct_enums.remove(0).1);
            }
        }
        hints
    }

    // ─── Type display ────────────────────────────────────────────────────────

    /// Returns a human-readable name for a type, used in type-mismatch error messages.
    pub(crate) fn display_type(ty: &Type) -> String {
        match ty {
            Type::Int    => "int".into(),
            Type::Uint   => "uint".into(),
            Type::Uint8  => "uint8".into(),
            Type::Int8   => "int8".into(),
            Type::Int16  => "int16".into(),
            Type::Int32  => "int32".into(),
            Type::Int64  => "int64".into(),
            Type::Int128 => "int128".into(),
            Type::Uint16 => "uint16".into(),
            Type::Uint32 => "uint32".into(),
            Type::Uint64 => "uint64".into(),
            Type::Uint128 => "uint128".into(),
            Type::Float32  => "float32".into(),
            Type::Float64  => "float64".into(),
            Type::Str    => "string".into(),
            Type::Bool   => "bool".into(),
            Type::Nil    => "nil".into(),
            Type::Void   => "void".into(),
            Type::Never  => "never".into(),
            Type::Named(n) => n.clone(),
            Type::Optional(inner) => format!("{}?", Self::display_type(inner)),
            Type::Array(e) => format!("[{}]", Self::display_type(e)),
            Type::ArrayN(e, n) => format!("[{}, {}]", Self::display_type(e), n),
            Type::ArrayNExpr(e, _) => format!("[{}, <expr>]", Self::display_type(e)),
            Type::LabeledArray(e, axes) => {
                let axes_str = axes.iter()
                    .map(|a| if a.size.is_some() { format!("{} = <expr>", a.label) } else { a.label.clone() })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{}, {}]", Self::display_type(e), axes_str)
            }
            Type::ConstInt(n) => n.to_string(),
            Type::Tuple(ts) => {
                let inner = ts.iter().map(Self::display_type).collect::<Vec<_>>().join(", ");
                format!("({})", inner)
            }
            Type::Dict(k, v) => format!("[{}: {}]", Self::display_type(k), Self::display_type(v)),
            Type::Set(e) => format!("{{{}}}", Self::display_type(e)),
            Type::Fn(ret, params, throws, task, req) => {
                let params_str = params.iter().map(Self::display_type).collect::<Vec<_>>().join(", ");
                let ret_str = ret.as_ref().map(|r| format!("{} ", Self::display_type(r))).unwrap_or_default();
                let throws_str = if *throws { " throws" } else { "" };
                let task_str = if *task { " task" } else { "" };
                let prefix = if *req { "req " } else { "" };
                format!("{}{}({}){}{}", prefix, ret_str, params_str, throws_str, task_str)
            }
            Type::Qualified(inner, qual) => {
                let qual_str = match qual {
                    OwnerQual::Owned        => "'owned".to_string(),
                    OwnerQual::Actor        => "'actor".to_string(),
                    OwnerQual::ActorTask    => "'task".to_string(),
                    OwnerQual::Guard        => "'guard".to_string(),
                    OwnerQual::GuardTask    => "'guard'task".to_string(),
                    OwnerQual::Shared       => "'shared".to_string(),
                    OwnerQual::Weak         => "'weak".to_string(),
                    OwnerQual::Inline       => "'inline".to_string(),
                    OwnerQual::Lifetime(lt) => format!("'{}", lt),
                    OwnerQual::Static       => "'static".to_string(),
                    OwnerQual::BorrowShared => "&shared".to_string(),
                    OwnerQual::BorrowOwned  => "&owned".to_string(),
                    OwnerQual::BorrowOption    => "?&".to_string(),
                    OwnerQual::BorrowOptionMut => "mut ?&".to_string(),
                    OwnerQual::BorrowWeak   => "&weak".to_string(),
                    OwnerQual::Borrow       => "&".to_string(),
                    OwnerQual::BorrowMut    => "var &".to_string(),
                    OwnerQual::GpuUnified   => "'gpu'unified".to_string(),
                    OwnerQual::GpuGlobal    => "'gpu'global".to_string(),
                    OwnerQual::GpuSurface   => "surface".to_string(),
                    OwnerQual::GpuActorGlobal => "'actor'global".to_string(),
                    OwnerQual::GpuActorUnified => "'actor'unified".to_string(),
                    OwnerQual::GpuLocal     => "'local".to_string(),
                    OwnerQual::GpuConst     => "'gpu'const".to_string(),
                    // `'new` (candidate-set qualifier, replaces the old bare tick) is
                    // represented as this exact Union shape — display it as `'new`
                    // rather than spelling out its members, matching source syntax.
                    OwnerQual::Union(members) if members.as_slice() == OwnerQual::NEW_MEMBERS =>
                        "'new".to_string(),
                    OwnerQual::Union(members) => {
                        let names: Vec<&str> = members.iter().map(|q| match q {
                            OwnerQual::Inline => "inline",
                            OwnerQual::Owned  => "owned",
                            OwnerQual::Shared => "shared",
                            OwnerQual::Actor  => "actor",
                            OwnerQual::Guard  => "guard",
                            _                 => "?",
                        }).collect();
                        format!("'{}", names.join("|"))
                    }
                };
                format!("{}{}", Self::display_type(inner), qual_str)
            }
            Type::TypeParam(name) => name.clone(),
            Type::Generic(name, args) => {
                if args.is_empty() {
                    name.clone()
                } else {
                    let args_str = args.iter().map(Self::display_type).collect::<Vec<_>>().join(", ");
                    format!("{}<{}>", name, args_str)
                }
            }
            Type::Dyn(inner) => format!("dyn {}", Self::display_type(inner)),
            Type::Impl(inner) => format!("impl {}", Self::display_type(inner)),
            Type::SelfAssoc(name) => name.clone(),
            Type::AssocOf(base, assoc) => {
                let base_str = match base.as_ref() {
                    Type::Named(n) | Type::Generic(n, _) => n.clone(),
                    other => Self::display_type(other),
                };
                format!("{}.{}", base_str, assoc)
            }
            Type::Mut(inner) => format!("mut {}", Self::display_type(inner)),
        }
    }

    // ─── Runtime type checking ───────────────────────────────────────────────

    /// Returns `true` if `val` is compatible with the **already-resolved** type `ty`.
    ///
    /// Rules:
    /// - Primitive types match their exact Value variants.
    /// - `Uint` also accepts `Int` (implicit coercion).
    /// - `Optional(T)` accepts `Nil` or any value that matches `T`.
    /// - `Qualified(T, _)` strips the qualifier and delegates to `T`.
    /// - `TypeParam` (still unresolved after stack lookup) accepts any value.
    /// - Collections are checked element-wise; empty collections always match.
    /// - `Fn` types only verify callability, not full signature.
    pub(crate) fn value_matches_type(&self, val: &Value, ty: &Type) -> bool {
        match ty {
            Type::Int    => matches!(val, Value::Int(_)),
            // Int is compatible with Uint only when non-negative (coerce_to_type handles the cast).
            // Negative Int values are rejected here so that `var uint x = -1` errors rather
            // than silently wrapping to 18446744073709551615.
            Type::Uint   => matches!(val, Value::Uint(_)) || matches!(val, Value::Int(n) if *n >= 0),
            // Uint8 accepts Int only when it fits in 0..=255 (coerce_to_type handles the cast).
            Type::Uint8  => matches!(val, Value::Uint8(_)) || matches!(val, Value::Int(n) if (0..=255).contains(n)),
            Type::Int8   => matches!(val, Value::Int8(_)) || matches!(val, Value::Int(n) if (i8::MIN as i64..=i8::MAX as i64).contains(n)),
            Type::Int16  => matches!(val, Value::Int16(_)) || matches!(val, Value::Int(n) if (i16::MIN as i64..=i16::MAX as i64).contains(n)),
            Type::Int32  => matches!(val, Value::Int32(_)) || matches!(val, Value::Int(n) if (i32::MIN as i64..=i32::MAX as i64).contains(n)),
            Type::Int64  => matches!(val, Value::Int64(_)) || matches!(val, Value::Int(_)),
            Type::Int128 => matches!(val, Value::Int128(_)) || matches!(val, Value::Int(_)),
            Type::Uint16 => matches!(val, Value::Uint16(_)) || matches!(val, Value::Int(n) if (0..=u16::MAX as i64).contains(n)),
            Type::Uint32 => matches!(val, Value::Uint32(_)) || matches!(val, Value::Int(n) if (0..=u32::MAX as i64).contains(n)),
            Type::Uint64 => matches!(val, Value::Uint64(_)) || matches!(val, Value::Int(n) if *n >= 0),
            Type::Uint128 => matches!(val, Value::Uint128(_)) || matches!(val, Value::Int(n) if *n >= 0),
            Type::Float32  => matches!(val, Value::Float32(_)),
            Type::Float64  => matches!(val, Value::Float64(_)),
            Type::Str    => matches!(val, Value::Str(_)),
            Type::Bool   => matches!(val, Value::Bool(_)),
            Type::Nil    => matches!(val, Value::Nil),
            Type::Void   => matches!(val, Value::Void),
            Type::Never  => false,
            Type::Optional(inner) => matches!(val, Value::Nil) || self.value_matches_type(val, inner),
            Type::Qualified(inner, _) => self.value_matches_type(val, inner),
            // `mut` has no Rust-level representation and runtime values carry no
            // permission tag — strip and delegate to the inner type.
            Type::Mut(inner) => self.value_matches_type(val, inner),
            Type::TypeParam(_) => true,
            Type::Named(name) => match val {
                Value::Channel { is_sender, .. } => match name.as_str() {
                    "Sender"   => *is_sender,
                    "Receiver" => !is_sender,
                    _ => false,
                },
                Value::Object(inner) => {
                    let type_name = inner.borrow().type_name.clone();
                    if &type_name == name { return true; }
                    // Check trait conformance: if `name` is a known trait, check if
                    // the object's struct declares or structurally satisfies it.
                    if self.traits.contains_key(name.as_str()) {
                        return self.object_conforms_to_trait(&type_name, name);
                    }
                    false
                }
                Value::EnumVariant { type_name, .. } => type_name == name,
                _ => false,
            },
            Type::Array(elem_ty) => match val {
                Value::Array(elems) => elems.is_empty() || elems.iter().all(|e| self.value_matches_type(e, elem_ty)),
                _ => false,
            },
            Type::ArrayN(elem_ty, n) => match val {
                Value::Array(elems) => elems.len() == *n && elems.iter().all(|e| self.value_matches_type(e, elem_ty)),
                _ => false,
            },
            // Runtime values carry no shape/label metadata (flat Value::Array, same as
            // Array/ArrayN) — check total length when it's a resolvable compile-time
            // constant (fixed-shape, mirrors ArrayN), otherwise permissive elementwise
            // check (dynamic-shape, mirrors Array). See docs/array-multidim-proposal.md.
            Type::LabeledArray(elem_ty, _) => match val {
                Value::Array(elems) => match ty.labeled_array_len() {
                    Some(n) => elems.len() as i64 == n && elems.iter().all(|e| self.value_matches_type(e, elem_ty)),
                    None => elems.is_empty() || elems.iter().all(|e| self.value_matches_type(e, elem_ty)),
                },
                _ => false,
            },
            Type::Set(elem_ty) => match val {
                Value::Set(elems) => elems.is_empty() || elems.iter().all(|e| self.value_matches_type(e, elem_ty)),
                _ => false,
            },
            Type::Dict(k_ty, v_ty) => match val {
                Value::Dict(pairs) => pairs.is_empty() || pairs.iter().all(|(k, v)| {
                    self.value_matches_type(k, k_ty) && self.value_matches_type(v, v_ty)
                }),
                _ => false,
            },
            Type::Tuple(types) => match val {
                Value::Tuple(elems) => {
                    elems.len() == types.len() &&
                    elems.iter().zip(types.iter()).all(|(e, t)| self.value_matches_type(e, t))
                }
                _ => false,
            },
            Type::Fn(..) => matches!(val, Value::Fn { .. } | Value::Closure { .. } | Value::NativeFn { .. }),
            // impl Trait — transparent at runtime, match against the inner trait type
            Type::Dyn(inner) | Type::Impl(inner) => self.value_matches_type(val, inner),
            Type::Generic(name, args) => match name.as_str() {
                // Result<T, E> — accept any Ok(v) / Err(e) EnumVariant; the inner
                // types are not checked at runtime (erasure), only the wrapper name.
                "Result" => matches!(val,
                    Value::EnumVariant { type_name, .. } if type_name == "Result"
                ),
                // Array<T>, Set<T>, Dict<K,V> should already be folded by resolve_type;
                // handle them defensively anyway.
                "Array" if args.len() == 1 =>
                    self.value_matches_type(val, &Type::Array(Box::new(args[0].clone()))),
                "Set" if args.len() == 1 =>
                    self.value_matches_type(val, &Type::Set(Box::new(args[0].clone()))),
                "Dict" if args.len() == 2 =>
                    self.value_matches_type(val, &Type::Dict(Box::new(args[0].clone()), Box::new(args[1].clone()))),
                "Future" => match val {
                    Value::Future(inner) =>
                        args.is_empty() || self.value_matches_type(inner, &args[0]),
                    _ => false,
                },
                // Channel endpoints: type parameter is erased at runtime (interpreter has
                // no type info on the channel's item type), so only the direction is checked.
                "Sender"   => matches!(val, Value::Channel { is_sender: true,  .. }),
                "Receiver" => matches!(val, Value::Channel { is_sender: false, .. }),
                // User-defined generic struct: check base name + specialize fields
                _ => match val {
                    Value::Object(inner_rc) => {
                        let (type_name, fields) = {
                            let inner = inner_rc.borrow();
                            (inner.type_name.clone(), inner.fields.clone())
                        };
                        if &type_name != name { return false; }
                        if args.is_empty() { return true; }
                        let struct_val = self.global.borrow().get(type_name.as_str());
                        if let Some(Value::Struct { decl, .. }) = struct_val {
                            if decl.type_params.len() != args.len() { return true; }
                            let bindings: HashMap<String, Type> = decl.type_params.iter()
                                .zip(args.iter())
                                .map(|(p, a)| (p.clone(), a.clone()))
                                .collect();
                            for field_decl in &decl.fields {
                                if let Some((_, fval)) = fields.iter().find(|(n, _)| n == &field_decl.name) {
                                    let resolved = Self::apply_type_bindings(&field_decl.ty, &bindings);
                                    if !self.value_matches_type(fval, &resolved) { return false; }
                                }
                            }
                            true
                        } else {
                            true  // struct not yet visible — accept
                        }
                    }
                    _ => false,
                },
            },
            // Associated type reference — accept any value at runtime (dynamically typed)
            Type::SelfAssoc(_) => true,
            Type::AssocOf(_, _) => true,
            // Const-generic types don't appear at runtime.
            Type::ArrayNExpr(_, _) | Type::ConstInt(_) => false,
        }
    }

    /// Simplified type matching for overload resolution — only checks the outermost type.
    /// Strips qualifiers, handles Named aliases for primitive types.
    pub(crate) fn value_matches_type_simple(&self, val: &Value, ty: &Type) -> bool {
        match ty {
            Type::Int    => matches!(val, Value::Int(_)),
            Type::Uint   => matches!(val, Value::Uint(_)),
            Type::Uint8  => matches!(val, Value::Uint8(_)),
            Type::Float64  => matches!(val, Value::Float64(_)),
            Type::Str    => matches!(val, Value::Str(_)),
            Type::Bool   => matches!(val, Value::Bool(_)),
            Type::Optional(_) => matches!(val, Value::Nil) || self.value_matches_type(val, ty),
            Type::Qualified(inner, _) => self.value_matches_type_simple(val, inner),
            Type::Named(name) => match name.as_str() {
                "int"    => matches!(val, Value::Int(_)),
                "uint"   => matches!(val, Value::Uint(_)),
                "uint8"  => matches!(val, Value::Uint8(_)),
                "float"  => matches!(val, Value::Float64(_)),
                "bool"   => matches!(val, Value::Bool(_)),
                "string" => matches!(val, Value::Str(_)),
                _ => self.value_matches_type(val, ty),
            },
            Type::Array(_) => matches!(val, Value::Array(_)),
            Type::Dict(_, _) => matches!(val, Value::Dict(_)),
            Type::Set(_)   => matches!(val, Value::Set(_)),
            _ => self.value_matches_type(val, ty),
        }
    }

    /// Returns true if the type named `type_name` conforms to `trait_name`.
    /// Checks (in order): explicit protocols list, conformance blocks,
    /// qualified methods, and structural (all methods present).
    /// Works for both structs and enums.
    pub(crate) fn object_conforms_to_trait(&self, type_name: &str, trait_name: &str) -> bool {
        let val = self.global.borrow().get(type_name);
        match val {
            Some(Value::Struct { decl, .. }) => {
                if decl.protocols.iter().any(|p| p == trait_name) { return true; }
                if decl.methods.iter().any(|m| m.qualifier.as_deref() == Some(trait_name)) { return true; }
                // Structural: all required method names present
                if let Some(trait_decl) = self.traits.get(trait_name) {
                    let method_names: std::collections::HashSet<&str> =
                        decl.methods.iter().map(|m| m.name.as_str()).collect();
                    return trait_decl.signatures.iter().all(|sig| method_names.contains(sig.name.as_str()));
                }
                false
            }
            Some(Value::EnumNamespace { methods, protocols, .. }) => {
                if protocols.iter().any(|p| p == trait_name) { return true; }
                if methods.iter().any(|m| m.qualifier.as_deref() == Some(trait_name)) { return true; }
                // Structural: all required method names present
                if let Some(trait_decl) = self.traits.get(trait_name) {
                    let method_names: std::collections::HashSet<&str> =
                        methods.iter().map(|m| m.name.as_str()).collect();
                    return trait_decl.signatures.iter().all(|sig| method_names.contains(sig.name.as_str()));
                }
                false
            }
            _ => false,
        }
    }

    /// Apply a map of type-param name → concrete type to a type expression.
    /// Pure static helper (no interpreter state needed).
    pub(crate) fn apply_type_bindings(ty: &Type, bindings: &HashMap<String, Type>) -> Type {
        match ty {
            Type::TypeParam(name) =>
                bindings.get(name.as_str()).cloned().unwrap_or_else(|| ty.clone()),
            Type::Named(name) =>
                bindings.get(name.as_str()).cloned().unwrap_or_else(|| ty.clone()),
            Type::Optional(inner) =>
                Type::Optional(Box::new(Self::apply_type_bindings(inner, bindings))),
            Type::Array(elem) =>
                Type::Array(Box::new(Self::apply_type_bindings(elem, bindings))),
            Type::Set(elem) =>
                Type::Set(Box::new(Self::apply_type_bindings(elem, bindings))),
            Type::Dict(k, v) => Type::Dict(
                Box::new(Self::apply_type_bindings(k, bindings)),
                Box::new(Self::apply_type_bindings(v, bindings)),
            ),
            Type::Tuple(elems) =>
                Type::Tuple(elems.iter().map(|e| Self::apply_type_bindings(e, bindings)).collect()),
            Type::Qualified(inner, qual) =>
                Type::Qualified(Box::new(Self::apply_type_bindings(inner, bindings)), qual.clone()),
            Type::Generic(name, args) =>
                Type::Generic(name.clone(), args.iter().map(|a| Self::apply_type_bindings(a, bindings)).collect()),
            other => other.clone(),
        }
    }

    // ─── Owned-collection rvalue extraction check ────────────────────────────

    /// Error if `expr` is a direct index/key access on an owned-element
    /// collection (`[T']`, `{K: T'}`).  Field access on top of an index
    /// (`v[i].age`) is NOT caught here — that is safe (ephemeral clone for
    /// reading a copy-field, or write-back via `assign`).
    pub(crate) fn check_no_owned_extract(expr: &Expr, env: &EnvRef, line: usize) -> Result<(), Signal> {
        if let ExprKind::Index(base, _) = &expr.kind {
            if let ExprKind::Var(name) = &base.kind {
                if env.borrow().is_owned_collection(name) {
                    return Err(err(
                        format!(
                            "cannot extract element from owned collection '{}'; \
                             use .remove() to transfer ownership, or mutate in-place via v[i].field = …",
                            name
                        ),
                        line,
                    ));
                }
            }
        }
        Ok(())
    }

    // ─── Enum field qualifier check ──────────────────────────────────────────

    /// Returns true if the type contains a disallowed qualifier for parametric
    /// enum fields (`'local` or `'shared`).  Checked recursively so that e.g.
    /// `[Counter'shared]` is also caught.
    /// Apply a Rust-style format specifier to a value.
    /// Grammar: [[fill]align][sign][#][0][width][.precision][type]
    /// align: < (left) | ^ (centre) | > (right, default for numbers)
    /// sign:  + (always show) | - (only negative, default)
    /// type:  x/X (hex) | b (binary) | o (octal) | e/E (scientific)
    ///        % (percent) | ? (debug) | (empty = Display)
    pub(crate) fn apply_format(val: Value, spec: &str, line: usize) -> Result<String, Signal> {
        let s = spec.trim();
        if s.is_empty() {
            return Ok(format!("{}", val));
        }
        let mut it = s.chars().peekable();

        // [[fill]align] — if second char is <, ^, > then first is the fill char
        let (fill, align): (char, Option<char>) = {
            let first = it.peek().copied().unwrap_or('\0');
            let second = {
                let mut tmp = it.clone();
                tmp.next();
                tmp.peek().copied().unwrap_or('\0')
            };
            if matches!(second, '<' | '^' | '>') {
                it.next(); it.next();
                (first, Some(second))
            } else if matches!(first, '<' | '^' | '>') {
                it.next();
                (' ', Some(first))
            } else {
                (' ', None)
            }
        };

        // [sign]
        let show_plus = match it.peek() {
            Some('+') => { it.next(); true  }
            Some('-') => { it.next(); false }
            _ => false,
        };

        // [#] alternate form (0x, 0b, 0o prefix)
        let alternate = if it.peek() == Some(&'#') { it.next(); true } else { false };

        // [0] zero-pad (only when no explicit fill/align)
        let zero_pad = if align.is_none() && it.peek() == Some(&'0') {
            let mut look = it.clone();
            look.next();
            if look.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                it.next(); true
            } else { false }
        } else { false };

        // [width]
        let width: Option<usize> = {
            let mut w = String::new();
            while it.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                w.push(it.next().unwrap());
            }
            w.parse().ok()
        };

        // [.precision]
        let precision: Option<usize> = if it.peek() == Some(&'.') {
            it.next();
            let mut p = String::new();
            while it.peek().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                p.push(it.next().unwrap());
            }
            Some(p.parse().unwrap_or(0))
        } else { None };

        // [type]
        let type_char = it.next().unwrap_or('\0');

        // ── helpers ──────────────────────────────────────────────────────────
        let to_int = |v: &Value| -> Result<i64, Signal> {
            match v {
                Value::Int(n)   => Ok(*n),
                Value::Uint(n)  => Ok(*n as i64),
                Value::Float64(f) => Ok(*f as i64),
                _ => Err(err(format!("format '{spec}' requires an integer"), line)),
            }
        };
        let to_float = |v: &Value| -> Result<f64, Signal> {
            match v {
                Value::Float64(f) => Ok(*f),
                Value::Int(n)   => Ok(*n as f64),
                Value::Uint(n)  => Ok(*n as f64),
                _ => Err(err(format!("format '{spec}' requires a number"), line)),
            }
        };

        // ── produce the unsized body ─────────────────────────────────────────
        let body: String = match type_char {
            'x' => {
                let n = to_int(&val)?;
                let s = if n < 0 { format!("-{:x}", (-n) as u64) } else { format!("{:x}", n as u64) };
                if alternate { format!("0x{s}") } else { s }
            }
            'X' => {
                let n = to_int(&val)?;
                let s = if n < 0 { format!("-{:X}", (-n) as u64) } else { format!("{:X}", n as u64) };
                if alternate { format!("0X{s}") } else { s }
            }
            'b' => {
                let n = to_int(&val)?;
                let s = if n < 0 { format!("-{:b}", (-n) as u64) } else { format!("{:b}", n as u64) };
                if alternate { format!("0b{s}") } else { s }
            }
            'o' => {
                let n = to_int(&val)?;
                let s = if n < 0 { format!("-{:o}", (-n) as u64) } else { format!("{:o}", n as u64) };
                if alternate { format!("0o{s}") } else { s }
            }
            'e' => {
                let f = to_float(&val)?;
                let prec = precision.unwrap_or(6);
                format!("{:.prec$e}", f)
            }
            'E' => {
                let f = to_float(&val)?;
                let prec = precision.unwrap_or(6);
                format!("{:.prec$E}", f)
            }
            '?' => {
                // debug repr: strings get quotes, others use Display
                match &val {
                    Value::Str(s) => format!("{s:?}"),
                    other => format!("{other}"),
                }
            }
            // default Display — with optional precision for floats
            _ => {
                if let Some(p) = precision {
                    // Precision on a numeric value = decimal places
                    if let Ok(f) = to_float(&val) {
                        format!("{:.1$}", f, p)
                    } else {
                        // Precision on strings = max char count
                        format!("{}", val).chars().take(p).collect()
                    }
                } else {
                    format!("{}", val)
                }
            }
        };

        // ── apply sign ───────────────────────────────────────────────────────
        let body = if show_plus && !body.starts_with('-') {
            match &val {
                Value::Int(n) if *n >= 0 => format!("+{body}"),
                Value::Uint(_) => format!("+{body}"),
                Value::Float64(f) if *f >= 0.0 => format!("+{body}"),
                _ => body,
            }
        } else { body };

        // ── apply width / alignment / padding ────────────────────────────────
        let result = if let Some(w) = width {
            let char_len = body.chars().count();
            if char_len >= w {
                body
            } else if zero_pad && align.is_none() {
                // Zero-pad: preserve leading sign char
                if body.starts_with('+') || body.starts_with('-') {
                    let (sign, rest) = body.split_at(1);
                    let pad: String = std::iter::repeat_n('0', w - char_len).collect();
                    format!("{sign}{pad}{rest}")
                } else {
                    let pad: String = std::iter::repeat_n('0', w - char_len).collect();
                    format!("{pad}{body}")
                }
            } else {
                let eff_align = align.unwrap_or('>');
                let pad_count = w - char_len;
                let pad: String = std::iter::repeat_n(fill, pad_count).collect();
                match eff_align {
                    '<' => format!("{body}{pad}"),
                    '^' => {
                        let left = pad_count / 2;
                        let right = pad_count - left;
                        let lp: String = std::iter::repeat_n(fill, left).collect();
                        let rp: String = std::iter::repeat_n(fill, right).collect();
                        format!("{lp}{body}{rp}")
                    }
                    _ => format!("{pad}{body}"),
                }
            }
        } else { body };

        Ok(result)
    }

    pub(crate) fn type_has_mutable_ref_qual(ty: &Type) -> bool {
        match ty {
            Type::Qualified(_, OwnerQual::Shared | OwnerQual::Guard) => true,
            Type::Qualified(inner, _) => Self::type_has_mutable_ref_qual(inner),
            Type::Optional(inner)
            | Type::Array(inner)
            | Type::Set(inner) => Self::type_has_mutable_ref_qual(inner),
            Type::Dict(k, v) =>
                Self::type_has_mutable_ref_qual(k) || Self::type_has_mutable_ref_qual(v),
            Type::Tuple(elems) => elems.iter().any(Self::type_has_mutable_ref_qual),
            Type::Generic(_, args) => args.iter().any(Self::type_has_mutable_ref_qual),
            _ => false,
        }
    }

    /// Verify that a parametric enum (one with type parameters) does not use
    /// `'local` or `'shared` qualifiers in its variant fields.
    pub(crate) fn check_enum_field_qualifiers(decl: &EnumDecl) -> Result<(), Signal> {
        if decl.type_params.is_empty() {
            return Ok(());  // plain enum — no restriction
        }
        for variant in &decl.variants {
            for field in &variant.fields {
                if Self::type_has_mutable_ref_qual(&field.ty) {
                    return Err(err(
                        format!(
                            "parametric enum '{}': variant field cannot use \
                             'shared qualifier (use 'new, 'copy or 'inline)",
                            decl.name
                        ),
                        decl.line,
                    ));
                }
            }
        }
        Ok(())
    }

    // ─── Ownership helpers ───────────────────────────────────────────────────

    /// Returns true if a type element qualifier is Owned.
    pub(crate) fn is_owned_qual(ty: &Type) -> bool {
        matches!(ty.without_mut(), Type::Qualified(_, OwnerQual::Owned))
    }

    /// Returns true if the type is a collection with exclusively-owned element type (`T'`).
    /// Returns true if a runtime value has copy semantics — moving it is a no-op.
    /// Only `T'` (owned exclusive) objects benefit from move; everything else copies freely.
    pub(crate) fn is_copy_value(val: &Value) -> bool {
        matches!(
            val,
            Value::Int(_)
                | Value::Uint(_)
                | Value::Float64(_)
                | Value::Bool(_)
                | Value::Str(_)
                | Value::Nil
                | Value::Void
                // Every sized/tagged numeric variant is Copy in the emitted Rust
                // (i8..i128/u8..u128/f32/f64 are all Copy) exactly like the
                // generic `Value::Int`/`Value::Uint`/`Value::Float64` above —
                // book.md documents ALL primitive numeric types as Copy. Before
                // this, only the generic untyped variants were exempted here,
                // so a `let t = n` on an `int64`/`int128` (etc.) scalar was
                // wrongly treated as a move (bug already fixed by including
                // the sized variants below).
                | Value::Int8(_)
                | Value::Int16(_)
                | Value::Int32(_)
                | Value::Int64(_)
                | Value::Int128(_)
                | Value::Uint8(_)
                | Value::Uint16(_)
                | Value::Uint32(_)
                | Value::Uint64(_)
                | Value::Uint128(_)
                | Value::Float32(_)
        )
    }

    pub(crate) fn type_has_owned_elems(ty: &Type) -> bool {
        match ty {
            Type::Array(e) | Type::Set(e) => Self::is_owned_qual(e),
            Type::Dict(_, v) => Self::is_owned_qual(v),
            _ => false,
        }
    }

    /// Returns true if a type annotation implies the variable is task-safe
    /// (qualified with 'rc, 'static, or 'copy, or is a primitive copy type).
    pub(crate) fn type_annotation_is_task_safe(ty: &Type) -> bool {
        ty.is_task_safe()
    }

    /// When a collection is declared with owned element type (`[T']`, `{T'}`, `{K: T'}`),
    /// invalidate the source variables of the elements in the initializer expression.
    pub(crate) fn invalidate_owned_collection_sources(ty: &Type, init: &Expr, env: &EnvRef) {
        match ty {
            Type::Array(elem) if Self::is_owned_qual(elem) => {
                if let ExprKind::Array(elems) = &init.kind {
                    for e in elems {
                        if let ExprKind::Var(name) = &e.kind {
                            env.borrow_mut().invalidate(name);
                        }
                    }
                }
            }
            Type::Set(elem) if Self::is_owned_qual(elem) => {
                if let ExprKind::Set(elems) = &init.kind {
                    for e in elems {
                        if let ExprKind::Var(name) = &e.kind {
                            env.borrow_mut().invalidate(name);
                        }
                    }
                }
            }
            Type::Dict(_, val_ty) if Self::is_owned_qual(val_ty) => {
                if let ExprKind::Dict(pairs) = &init.kind {
                    for (_, v) in pairs {
                        if let ExprKind::Var(name) = &v.kind {
                            env.borrow_mut().invalidate(name);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // ─── Task capture safety ─────────────────────────────────────────────────

    /// Collect all Var names referenced in an expression (recursively).
    pub(crate) fn collect_vars_expr(expr: &Expr, out: &mut Vec<String>) {
        match &expr.kind {
            ExprKind::Var(name) => out.push(name.clone()),
            ExprKind::BinOp(_, l, r) => { Self::collect_vars_expr(l, out); Self::collect_vars_expr(r, out); }
            ExprKind::UnaryOp(_, e) => Self::collect_vars_expr(e, out),
            ExprKind::Assign(t, v) | ExprKind::QuestionAssign(t, v) => { Self::collect_vars_expr(t, out); Self::collect_vars_expr(v, out); }
            ExprKind::Field(e, _) | ExprKind::OptionalField(e, _) => Self::collect_vars_expr(e, out),
            ExprKind::Index(e, i) => { Self::collect_vars_expr(e, out); Self::collect_vars_expr(i, out); }
            ExprKind::Call(f, args) => {
                Self::collect_vars_expr(f, out);
                for a in args { Self::collect_vars_expr(&a.value, out); }
            }
            ExprKind::MethodCall(e, _, args) | ExprKind::OptionalMethodCall(e, _, args) => {
                Self::collect_vars_expr(e, out);
                for a in args { Self::collect_vars_expr(&a.value, out); }
            }
            ExprKind::Array(elems) | ExprKind::Tuple(elems) | ExprKind::Set(elems) => {
                for e in elems { Self::collect_vars_expr(e, out); }
            }
            ExprKind::Dict(pairs) => {
                for (k, v) in pairs { Self::collect_vars_expr(k, out); Self::collect_vars_expr(v, out); }
            }
            ExprKind::Else(e, d) | ExprKind::TryElse(e, d) => {
                Self::collect_vars_expr(e, out); Self::collect_vars_expr(d, out);
            }
            ExprKind::TryElseBlock(try_stmts, else_stmts) => {
                Self::collect_vars_stmts(try_stmts, out);
                Self::collect_vars_stmts(else_stmts, out);
            }
            ExprKind::Cast(e, _) => Self::collect_vars_expr(e, out),
            ExprKind::Range { start, end, .. } => {
                Self::collect_vars_expr(start, out); Self::collect_vars_expr(end, out);
            }
            ExprKind::Task(e) => Self::collect_vars_expr(e, out),
            ExprKind::TaskWithTimeout(dur, body) => {
                Self::collect_vars_expr(dur, out);
                Self::collect_vars_expr(body, out);
            }
            ExprKind::MacroCall { args, .. } => {
                for a in args { Self::collect_vars_expr(a, out); }
            }
            ExprKind::Block(stmts) | ExprKind::Do(stmts) => Self::collect_vars_stmts(stmts, out),
            ExprKind::Loop(s) => Self::collect_vars_stmts(&s.body, out),
            ExprKind::If(s) => {
                for (cond, body) in &s.branches {
                    Self::collect_vars_expr(cond, out);
                    Self::collect_vars_stmts(body, out);
                }
                if let Some(e) = &s.else_body { Self::collect_vars_stmts(e, out); }
            }
            ExprKind::Match(s) => {
                Self::collect_vars_expr(&s.subject, out);
                for arm in &s.arms {
                    match &arm.body {
                        MatchBody::Expr(e) => Self::collect_vars_expr(e, out),
                        MatchBody::Block(stmts) => Self::collect_vars_stmts(stmts, out),
                    }
                }
            }
            ExprKind::Closure(_, _, body, _, _) => {
                match body {
                    ClosureBody::Expr(e) => Self::collect_vars_expr(e, out),
                    ClosureBody::Block(stmts) => Self::collect_vars_stmts(stmts, out),
                }
            }
            ExprKind::StringInterp(segs) => {
                for seg in segs {
                    match seg {
                        StringSegment::Expr(e) | StringSegment::FormattedExpr(e, _) => {
                            Self::collect_vars_expr(e, out);
                        }
                        StringSegment::Lit(_) => {}
                    }
                }
            }
            // `f<T>(args)` — type args carry no variable refs, recurse callee + args
            ExprKind::GenericCall(callee, _type_args, args) => {
                Self::collect_vars_expr(callee, out);
                for a in args { Self::collect_vars_expr(&a.value, out); }
            }
            // `lhs |> f(args)` — recurse lhs + args
            ExprKind::Pipe(lhs, _, args) => {
                Self::collect_vars_expr(lhs, out);
                for a in args { Self::collect_vars_expr(&a.value, out); }
            }
            // `join [f1, f2, …]` — recurse all handles
            ExprKind::JoinAll(exprs) => {
                for e in exprs { Self::collect_vars_expr(e, out); }
            }
            _ => {}
        }
    }

    pub(crate) fn collect_vars_stmts(stmts: &[Stmt], out: &mut Vec<String>) {
        for s in stmts {
            match s {
                Stmt::Expr(e) => Self::collect_vars_expr(e, out),
                Stmt::Let(l) => { if let Some(v) = &l.value { Self::collect_vars_expr(v, out); } }
                Stmt::Return(r) => { if let Some(e) = &r.value { Self::collect_vars_expr(e, out); } }
                Stmt::Throw(t) => { if let Some(e) = &t.value { Self::collect_vars_expr(e, out); } }
                Stmt::If(i) => {
                    for (cond, body) in &i.branches {
                        Self::collect_vars_expr(cond, out);
                        Self::collect_vars_stmts(body, out);
                    }
                    if let Some(e) = &i.else_body { Self::collect_vars_stmts(e, out); }
                }
                Stmt::While(w) => { Self::collect_vars_expr(&w.condition, out); Self::collect_vars_stmts(&w.body, out); }
                Stmt::For(f) => { Self::collect_vars_expr(&f.iterable, out); Self::collect_vars_stmts(&f.body, out); }
                Stmt::Try(t) => {
                    Self::collect_vars_stmts(&t.body, out);
                    for c in &t.catch_clauses { Self::collect_vars_stmts(&c.body, out); }
                }
                Stmt::Fn(f) => Self::collect_vars_stmts(&f.body, out),
                _ => {}
            }
        }
    }

    /// Returns true if this runtime value is a heap type that's unsafe to share across tasks
    /// without an explicit qualifier ('rc, 'static, 'copy).
    pub(crate) fn is_unqualified_heap_type(val: &Value) -> bool {
        matches!(val, Value::Array(_) | Value::Dict(_) | Value::Set(_) | Value::Object(_))
    }

    /// Check that a task body does not capture unqualified heap types from the enclosing scope.
    /// Variables explicitly declared with 'rc, 'static, or 'copy bypass this check.
    pub(crate) fn check_task_captures(expr: &Expr, env: &EnvRef, line: usize) -> Result<(), Signal> {
        let mut vars = Vec::new();
        Self::collect_vars_expr(expr, &mut vars);
        vars.sort();
        vars.dedup();
        for name in &vars {
            // Skip variables that were declared with a task-safe qualifier
            if env.borrow().is_task_safe_var(name) {
                continue;
            }
            if let Some(val) = env.borrow().get(name) {
                if Self::is_unqualified_heap_type(&val) {
                    return Err(err(
                        format!(
                            "task cannot capture '{}' ({}): annotate with 'rc or 'static to allow sharing",
                            name, val.type_name()
                        ),
                        line,
                    ));
                }
            }
        }
        Ok(())
    }

    /// Collect names of variables captured by the task body that are 'owned (move semantics).
    /// These will be invalidated after the task takes ownership.
    pub(crate) fn collect_owned_task_captures(expr: &Expr, env: &EnvRef) -> Vec<String> {
        let mut vars = Vec::new();
        Self::collect_vars_expr(expr, &mut vars);
        vars.sort();
        vars.dedup();
        vars.into_iter()
            .filter(|name| {
                // Only invalidate vars that are owned-qualified in the env
                env.borrow().is_owned_var(name)
            })
            .collect()
    }

    // ─── Expressions ────────────────────────────────────────────────────────

}
