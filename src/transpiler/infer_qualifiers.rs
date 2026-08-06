use super::Transpiler;
use crate::ast::{BindingKind, Expr, ExprKind, MatchBody, OwnerQual, Stmt, Type};
use super::helpers::collect_var_names;

impl Transpiler {
    /// Pre-pass: walk a function body and populate `inferred_qualifiers`.
    ///
    /// Each anonymous local variable starts as a candidate for all qualifiers:
    /// {Stack, Owned, Shared, Actor, Guard}. Every usage signal eliminates
    /// incompatible qualifiers from the candidate set (constraint elimination).
    ///
    /// Resolution at the end of the pass:
    /// - exactly 1 candidate remaining → that qualifier is inferred
    /// - 0 candidates → conflict error (no qualifier satisfies all constraints)
    /// - >1 candidates → no inference (size-based fallback applies at emit time)
    ///
    /// Alias rule: `let y = x` records `y` as an alias of `x`. Constraints applied
    /// to either member are propagated to the whole group.
    pub(crate) fn infer_qualifiers(&mut self, stmts: &[Stmt]) {
        self.inferred_qualifiers.clear();
        self.task_method_call_vars.clear();

        let mut alias_of: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        let mut anonymous_vars: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut var_struct_types: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        let mut mut_bindings: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut tick_bindings: std::collections::HashSet<String> = std::collections::HashSet::new();
        // Union-typed params: maps name → initial candidate set (the Union members).
        let mut union_initial: std::collections::HashMap<String, Vec<OwnerQual>> = std::collections::HashMap::new();
        // Params eligible for auto-ref inference (bare named type, in type_sizes).
        let mut auto_ref_param_vars: std::collections::HashSet<String> = std::collections::HashSet::new();
        // Params that received a qualifier demand or storage signal during the walk.
        // If a param is NOT in this set after the walk, it is a candidate for auto-ref.
        let mut has_qualifier_constraint: std::collections::HashSet<String> = std::collections::HashSet::new();

        let return_qual = self.fn_return_ty.as_ref().and_then(qual_of_type);

        // Seed anonymous_vars with unqualified and Union-qualified parameters so that
        // body usage signals can constrain them just like local let-bindings.
        for (name, ty) in &self.fn_current_params {
            match ty {
                // Bare T parameter — full candidate set.
                // Only include parameters whose type is a user-defined struct or enum
                // (present in type_sizes). Primitives, traits, type aliases, fn-type aliases,
                // and type parameters are excluded: the fallback would infer 'stack and
                // emit_param would wrap them incorrectly (Addable'stack → "Addable" instead
                // of impl Addable, Pt'stack bypasses the non-fn alias expansion, etc.).
                Type::Named(n) if self.type_sizes.contains_key(n.as_str())
                    || self.all_struct_types.contains(n.as_str()) => {
                    anonymous_vars.insert(name.clone());
                    // Track the struct/enum type name so resolve_fallback knows it's a user struct type.
                    var_struct_types.insert(name.clone(), n.clone());
                    // Eligible for auto-ref inference — free functions only, known-size types only.
                    // Types with dynamic fields (in all_struct_types but not type_sizes) are excluded:
                    // they do not benefit from borrow inference and may be actor-source types.
                    if !self.in_struct_method && self.type_sizes.contains_key(n.as_str()) {
                        auto_ref_param_vars.insert(name.clone());
                    }
                }
                // Bare [T] / {K=V} / {T} parameter — array/dict/set. Same auto-ref treatment
                // as bare struct params: free functions only (never struct methods — mirrors
                // the deliberately-unresolved struct-method case above). Unlike structs, these
                // have no entry in type_sizes/all_struct_types (they're built-in collection
                // types, not user-defined), so eligibility doesn't gate on that — only on
                // being a free-function param at all.
                Type::Array(_) | Type::Dict(_, _) | Type::Set(_) => {
                    anonymous_vars.insert(name.clone());
                    if !self.in_struct_method {
                        auto_ref_param_vars.insert(name.clone());
                    }
                }
                // T? parameter — bare optional struct: eligible for qualifier inference.
                // Optional params are never auto-ref (Option<&T> is not useful).
                Type::Optional(inner) => {
                    if let Type::Named(n) = inner.as_ref() {
                        if self.type_sizes.contains_key(n.as_str()) || self.qualified_struct_types.contains(n.as_str()) {
                            anonymous_vars.insert(name.clone());
                            var_struct_types.insert(name.clone(), n.clone());
                        }
                    }
                }
                // T' parameter — indirection-only candidate set.
                Type::Qualified(_, OwnerQual::Owned) => {
                    anonymous_vars.insert(name.clone());
                    tick_bindings.insert(name.clone());
                }
                // T'<group> parameter — Union members as candidate set.
                Type::Qualified(_, OwnerQual::Union(members)) => {
                    anonymous_vars.insert(name.clone());
                    union_initial.insert(name.clone(), members.clone());
                }
                _ => {}
            }
        }

        for stmt in stmts {
            collect_anonymous_vars(stmt, &mut anonymous_vars, &mut alias_of, &mut var_struct_types, &mut mut_bindings, &mut tick_bindings);
        }

        // Each anonymous variable starts as a candidate for every qualifier.
        // T' → indirection-only; T'<group> → Union members; bare T → full set.
        let mut candidates: std::collections::HashMap<String, Vec<OwnerQual>> = anonymous_vars
            .iter()
            .map(|name| {
                let quals = if let Some(initial) = union_initial.get(name.as_str()) {
                    initial.clone()
                } else if tick_bindings.contains(name.as_str()) {
                    indirection_qualifiers()
                } else {
                    all_qualifiers()
                };
                (name.clone(), quals)
            })
            .collect();

        // `mut` binding → mutation signal at declaration site: eliminates Shared.
        for var_name in &mut_bindings {
            constrain_candidates(
                &mut candidates, var_name,
                &[OwnerQual::Stack, OwnerQual::Owned, OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask],
                &alias_of,
            );
        }

        // Actor-source type constraint: if a type T is known to be produced by an 'actor-returning
        // function (recorded in actor_source_types during pre_scan), immediately constrain bare T
        // params to {Actor, Guard}. This enables automatic inference without manual annotation.
        for var_name in anonymous_vars.iter() {
            if let Some(struct_name) = var_struct_types.get(var_name.as_str()) {
                if self.actor_source_types.contains(struct_name.as_str()) {
                    constrain_candidates(
                        &mut candidates, var_name,
                        &[OwnerQual::Actor, OwnerQual::Guard],
                        &alias_of,
                    );
                    has_qualifier_constraint.insert(var_name.clone());
                }
            }
        }

        // Pre-pass: collect local variables assigned from 'actor-returning calls.
        // Recursive so nested let-bindings (inside if/for/while bodies) are found.
        self.infer_local_actor_vars.clear();
        fn collect_actor_lets(transpiler: &Transpiler, stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
            for stmt in stmts {
                match stmt {
                    Stmt::Let(s) => {
                        if let Some(val) = &s.value {
                            if transpiler.expr_returns_actor_qual(val) {
                                out.insert(s.name.clone());
                            }
                        }
                    }
                    Stmt::If(s) => {
                        for (_, body) in &s.branches { collect_actor_lets(transpiler, body, out); }
                        if let Some(eb) = &s.else_body { collect_actor_lets(transpiler, eb, out); }
                    }
                    Stmt::IfLet(s) => {
                        collect_actor_lets(transpiler, &s.then_body, out);
                        for branch in &s.elif_branches { collect_actor_lets(transpiler, &branch.body, out); }
                        if let Some(eb) = &s.else_body { collect_actor_lets(transpiler, eb, out); }
                    }
                    Stmt::While(s) => { collect_actor_lets(transpiler, &s.body, out); }
                    Stmt::For(s) => { collect_actor_lets(transpiler, &s.body, out); }
                    Stmt::Match(s) => {
                        for arm in &s.arms {
                            if let crate::ast::MatchBody::Block(body) = &arm.body {
                                collect_actor_lets(transpiler, body, out);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut local_actor_vars = std::collections::HashSet::new();
        collect_actor_lets(self, stmts, &mut local_actor_vars);
        self.infer_local_actor_vars = local_actor_vars;

        // Direct 'actor constraint: any anonymous var (local or param) that is known to hold an
        // 'actor value (from infer_local_actor_vars) is constrained to {Actor, Guard} immediately.
        for var_name in self.infer_local_actor_vars.iter() {
            if anonymous_vars.contains(var_name.as_str()) {
                constrain_candidates(
                    &mut candidates, var_name,
                    &[OwnerQual::Actor, OwnerQual::Guard],
                    &alias_of,
                );
                has_qualifier_constraint.insert(var_name.clone());
            }
        }

        for stmt in stmts {
            self.walk_stmt_for_qualifiers(
                stmt, &anonymous_vars, &var_struct_types, &alias_of,
                return_qual.as_ref(), &mut candidates, &auto_ref_param_vars,
                &mut has_qualifier_constraint,
            );
        }

        // Tail-expression inference: bare variable as last expression inherits return qualifier.
        if let Some(ref rq) = return_qual {
            if let Some(Stmt::Expr(e)) = stmts.iter().rev().find(|s| !matches!(s, Stmt::Defer(_))) {
                if let ExprKind::Var(name) = &e.kind {
                    if anonymous_vars.contains(name.as_str()) {
                        constrain_candidates(&mut candidates, name, std::slice::from_ref(rq), &alias_of);
                        if auto_ref_param_vars.contains(name.as_str()) {
                            has_qualifier_constraint.insert(name.clone());
                        }
                    }
                }
            }
        }

        // Resolve candidates → inferred_qualifiers.
        for (var_name, remaining) in &candidates {
            // Only report for roots (not aliases) to avoid duplicate errors.
            let is_alias = alias_of.contains_key(var_name.as_str());
            // If both a plain qualifier and its 'task variant survived elimination, pick
            // one based on whether a task-declared method was called on this variable.
            let mut remaining: Vec<OwnerQual> = remaining.clone();
            disambiguate_task_variant(&mut remaining, self.task_method_call_vars.contains(var_name.as_str()));
            match remaining.len() {
                0 if !is_alias => {
                    let line = self.fn_current_param_lines.get(var_name.as_str()).copied().unwrap_or(0);
                    let loc = if line > 0 { format!(" line {}", line) } else { String::new() };
                    eprintln!(
                        "error{}: `{}` has no valid qualifier — usage constraints are incompatible\n  \
                         fix: annotate `{}` explicitly",
                        loc, var_name, var_name
                    );
                }
                1 => {
                    self.inferred_qualifiers.insert(var_name.clone(), remaining[0].clone());
                }
                _ => {
                    // Pre-fallback: universal borrow inference for bare parameters.
                    // If the param had no qualifier demand or storage signal during the walk,
                    // resolve to Counter& (immutable) or mut Counter& (mutable).
                    if auto_ref_param_vars.contains(var_name.as_str())
                        && !has_qualifier_constraint.contains(var_name.as_str())
                    {
                        let is_mut = self.fn_current_params_mut.contains(var_name.as_str());
                        let qual = if is_mut { OwnerQual::BorrowMut } else { OwnerQual::Borrow };
                        self.inferred_qualifiers.insert(var_name.clone(), qual);
                        continue;
                    }
                    // Multiple candidates remaining: apply priority-ordered fallback.
                    let type_size = var_struct_types.get(var_name.as_str())
                        .and_then(|tn| self.type_sizes.get(tn.as_str()))
                        .copied();
                    if let Some(q) = resolve_fallback(
                        &remaining, false, type_size, self.config.stack_auto_bytes,
                    ) {
                        self.inferred_qualifiers.insert(var_name.clone(), q);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_stmt_for_qualifiers(
        &mut self,
        stmt: &Stmt,
        anonymous_vars: &std::collections::HashSet<String>,
        var_struct_types: &std::collections::HashMap<String, String>,
        alias_of: &std::collections::HashMap<String, String>,
        return_qual: Option<&OwnerQual>,
        candidates: &mut std::collections::HashMap<String, Vec<OwnerQual>>,
        auto_ref_param_vars: &std::collections::HashSet<String>,
        has_qualifier_constraint: &mut std::collections::HashSet<String>,
    ) {
        match stmt {
            Stmt::Let(s) => {
                if let Some(val) = &s.value {
                    self.walk_expr_for_qualifiers(val, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                }
            }
            Stmt::Expr(e) => {
                self.walk_expr_for_qualifiers(e, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
            }
            Stmt::Return(r) => {
                if let Some(e) = &r.value {
                    if let (Some(rq), ExprKind::Var(name)) = (return_qual, &e.kind) {
                        if anonymous_vars.contains(name.as_str()) {
                            constrain_candidates(candidates, name, std::slice::from_ref(rq), alias_of);
                            if auto_ref_param_vars.contains(name.as_str()) {
                                has_qualifier_constraint.insert(name.clone());
                            }
                        }
                    }
                    self.walk_expr_for_qualifiers(e, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                }
            }
            Stmt::If(s) => {
                for (cond, body) in &s.branches {
                    self.walk_expr_for_qualifiers(cond, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                    for st in body {
                        self.walk_stmt_for_qualifiers(st, anonymous_vars, var_struct_types, alias_of, return_qual, candidates, auto_ref_param_vars, has_qualifier_constraint);
                    }
                }
                if let Some(else_body) = &s.else_body {
                    for st in else_body {
                        self.walk_stmt_for_qualifiers(st, anonymous_vars, var_struct_types, alias_of, return_qual, candidates, auto_ref_param_vars, has_qualifier_constraint);
                    }
                }
            }
            Stmt::While(s) => {
                self.walk_expr_for_qualifiers(&s.condition, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                for st in &s.body {
                    self.walk_stmt_for_qualifiers(st, anonymous_vars, var_struct_types, alias_of, return_qual, candidates, auto_ref_param_vars, has_qualifier_constraint);
                }
            }
            Stmt::For(s) => {
                self.walk_expr_for_qualifiers(&s.iterable, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                for st in &s.body {
                    self.walk_stmt_for_qualifiers(st, anonymous_vars, var_struct_types, alias_of, return_qual, candidates, auto_ref_param_vars, has_qualifier_constraint);
                }
            }
            Stmt::Match(s) => {
                // A param used as a match subject must be owned (taken by value) so that
                // bound variables in arm patterns have their concrete field types, not references.
                // Suppress auto-ref inference for such params.
                if let ExprKind::Var(vname) = &s.subject.kind {
                    if auto_ref_param_vars.contains(vname.as_str()) {
                        has_qualifier_constraint.insert(vname.clone());
                    }
                }
                self.walk_expr_for_qualifiers(&s.subject, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                for arm in &s.arms {
                    if let Some(guard) = &arm.guard {
                        self.walk_expr_for_qualifiers(guard, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                    }
                    match &arm.body {
                        MatchBody::Expr(e) => {
                            self.walk_expr_for_qualifiers(e, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                        }
                        MatchBody::Block(stmts) => {
                            for st in stmts {
                                self.walk_stmt_for_qualifiers(st, anonymous_vars, var_struct_types, alias_of, return_qual, candidates, auto_ref_param_vars, has_qualifier_constraint);
                            }
                        }
                    }
                }
            }
            // Destructuring `let Some(x) = param.field else { ... }` requires moving out of
            // `param.field`. If `param` is auto-ref inferred as `&T`, this fails (can't move
            // out of a shared reference). Suppress auto-ref for the param.
            Stmt::LetDestructure(s) => {
                if let ExprKind::Field(obj, _) = &s.value.kind {
                    if let ExprKind::Var(vname) = &obj.kind {
                        if auto_ref_param_vars.contains(vname.as_str()) {
                            has_qualifier_constraint.insert(vname.clone());
                        }
                    }
                }
                self.walk_expr_for_qualifiers(&s.value, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
            }
            // if-let patterns: `let Some(x) = param.field` in a CondClause moves out of
            // `param.field` — suppress auto-ref on the root param variable.
            Stmt::IfLet(s) => {
                for clause in &s.clauses {
                    self.walk_cond_clause_for_qualifiers(clause, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                }
                for st in &s.then_body {
                    self.walk_stmt_for_qualifiers(st, anonymous_vars, var_struct_types, alias_of, return_qual, candidates, auto_ref_param_vars, has_qualifier_constraint);
                }
                for branch in &s.elif_branches {
                    for clause in &branch.clauses {
                        self.walk_cond_clause_for_qualifiers(clause, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                    }
                    for st in &branch.body {
                        self.walk_stmt_for_qualifiers(st, anonymous_vars, var_struct_types, alias_of, return_qual, candidates, auto_ref_param_vars, has_qualifier_constraint);
                    }
                }
                if let Some(else_body) = &s.else_body {
                    for st in else_body {
                        self.walk_stmt_for_qualifiers(st, anonymous_vars, var_struct_types, alias_of, return_qual, candidates, auto_ref_param_vars, has_qualifier_constraint);
                    }
                }
            }
            // guard let Some(x) = param.field else: — same ownership constraint as IfLet/LetDestructure.
            Stmt::Guard(s) => {
                if let crate::ast::GuardCond::Clauses(clauses) = &s.cond {
                    for clause in clauses {
                        self.walk_cond_clause_for_qualifiers(clause, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                    }
                }
            }
            _ => {}
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_expr_for_qualifiers(
        &mut self,
        expr: &Expr,
        anonymous_vars: &std::collections::HashSet<String>,
        var_struct_types: &std::collections::HashMap<String, String>,
        alias_of: &std::collections::HashMap<String, String>,
        candidates: &mut std::collections::HashMap<String, Vec<OwnerQual>>,
        auto_ref_param_vars: &std::collections::HashSet<String>,
        has_qualifier_constraint: &mut std::collections::HashSet<String>,
    ) {
        match &expr.kind {
            ExprKind::Call(callee, args) => {
                self.walk_expr_for_qualifiers(callee, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                // Detect struct-constructor call: `Foo(field: expr, ...)`.
                // In boring, `field: expr` inside a call is a single-param closure `(field): expr`
                // that serves as a labeled-arg shorthand for struct construction.
                // These are NOT real closures — their bodies should be walked without capture constraints.
                let is_struct_ctor = if let ExprKind::Var(fn_name) = &callee.kind {
                    fn_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                        && self.struct_fields.contains_key(fn_name.as_str())
                } else { false };
                for arg in args {
                    // For struct-ctor calls, unwrap single-param labeled-arg closures.
                    let walk_target: &Expr = if is_struct_ctor {
                        if let ExprKind::Closure(params, _, body, _, _) = &arg.value.kind {
                            if params.len() == 1 {
                                if let crate::ast::ClosureBody::Expr(e) = body { e.as_ref() } else { &arg.value }
                            } else { &arg.value }
                        } else { &arg.value }
                    } else { &arg.value };
                    self.walk_expr_for_qualifiers(walk_target, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                }
                // Call site with a concrete qualifier demand: intersect to the compatible set.
                // For 'shared/'actor/'guard demands, 'stack and 'heap are also compatible
                // because a plain T or Box<T> can be wrapped at the call site.
                if let ExprKind::Var(fn_name) = &callee.kind {
                    let param_types = self.fn_sigs.get(fn_name.as_str()).cloned();
                    if let Some(param_types) = param_types {
                        for (i, arg) in args.iter().enumerate() {
                            let Some(param_ty) = param_types.get(i) else { continue };
                            let Some(demanded) = qual_of_type(param_ty) else { continue };
                            let ExprKind::Var(var_name) = &arg.value.kind else { continue };
                            if anonymous_vars.contains(var_name.as_str()) {
                                // Optional params require exact qualifier match: Option<T> cannot be
                                // auto-coerced to Option<Arc<Mutex<T>>> at the call site.
                                let compatible = if matches!(param_ty, Type::Optional(_)) {
                                    vec![demanded.clone()]
                                } else {
                                    coercible_from(demanded.clone())
                                };
                                constrain_candidates(candidates, var_name, &compatible, alias_of);
                                // A concrete qualifier demand (not Borrow/BorrowMut) is a
                                // qualifier-demand signal: auto-ref inference does not apply.
                                if auto_ref_param_vars.contains(var_name.as_str())
                                    && !matches!(demanded, OwnerQual::Borrow | OwnerQual::BorrowMut) {
                                        has_qualifier_constraint.insert(var_name.clone());
                                    }
                            }
                        }
                    }
                    // Struct constructor call: `Env(parent = x, ...)` — constrain named args
                    // by the corresponding struct field's declared qualifier.
                    let is_struct_ctor = fn_name.chars().next()
                        .map(|c| c.is_uppercase()).unwrap_or(false)
                        && self.struct_fields.contains_key(fn_name.as_str());
                    if is_struct_ctor {
                        let fields = self.struct_fields.get(fn_name.as_str()).cloned().unwrap_or_default();
                        for arg in args {
                            // Resolve both explicit label (`field= x`) and closure-style (`field: x`).
                            let (field_name_opt, val_expr): (Option<&String>, &Expr) =
                                if let Some(lbl) = &arg.label {
                                    (Some(lbl), &arg.value)
                                } else if let ExprKind::Closure(params, _, body, _, _) = &arg.value.kind {
                                    if params.len() == 1 {
                                        let field_name = &params[0].name;
                                        if let crate::ast::ClosureBody::Expr(e) = body {
                                            (Some(field_name), e.as_ref())
                                        } else { (None, &arg.value) }
                                    } else { (None, &arg.value) }
                                } else { (None, &arg.value) };
                            let ExprKind::Var(var_name) = &val_expr.kind else { continue };
                            if !anonymous_vars.contains(var_name.as_str()) { continue; }
                            let Some(field_name) = field_name_opt else { continue };
                            let Some((_, field_ty)) = fields.iter().find(|(n, _)| n == field_name) else { continue };
                            let Some(demanded) = qual_of_type(field_ty) else { continue };
                            let compatible = if matches!(field_ty, Type::Optional(_)) {
                                vec![demanded.clone()]
                            } else {
                                coercible_from(demanded.clone())
                            };
                            constrain_candidates(candidates, var_name, &compatible, alias_of);
                        }
                    }
                }
            }
            ExprKind::MethodCall(obj, method, args) => {
                self.walk_expr_for_qualifiers(obj, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                for arg in args {
                    self.walk_expr_for_qualifiers(&arg.value, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                }
                // def (mutating) method call: variable must support direct mutation.
                // Eliminates Shared from the candidate set.
                // For auto-ref params declared without `mut`: this is a compile error.
                if let ExprKind::Var(var_name) = &obj.kind {
                    if anonymous_vars.contains(var_name.as_str()) {
                        let is_req = if let Some(struct_name) = var_struct_types.get(var_name.as_str()) {
                            self.struct_req_methods.contains(&format!("{}::{}", struct_name, method))
                        } else {
                            // Not a user struct — check whether it's a bare array/dict/set
                            // param. Built-in collection methods are read-only unless listed
                            // in MUTATING_COLLECTION_METHODS (mirrors the ACTOR_FIELD_MUTATING
                            // check in emit_methods.rs), so e.g. `.len()`/`.contains()` on an
                            // unqualified `[T]`/`{K=V}`/`{T}` param no longer blocks auto-ref.
                            matches!(
                                self.fn_current_params.get(var_name.as_str()),
                                Some(Type::Array(_) | Type::Dict(_, _) | Type::Set(_))
                            ) && !super::helpers::MUTATING_COLLECTION_METHODS.contains(&method.as_str())
                        };
                        if !is_req {
                            let is_auto_ref_param = auto_ref_param_vars.contains(var_name.as_str());
                            let is_mut_param = self.fn_current_params_mut.contains(var_name.as_str());
                            // Error: def call on immutable auto-ref parameter.
                            if is_auto_ref_param && !is_mut_param {
                                let line = self.fn_current_param_lines.get(var_name.as_str()).copied().unwrap_or(0);
                                let loc = if line > 0 { format!(" line {}", line) } else { String::new() };
                                eprintln!(
                                    "error{}: parameter `{}` is immutable but `{}` is a `def` method \
                                     — declare `mut {} n`",
                                    loc, var_name, method, var_name
                                );
                            }
                            constrain_candidates(
                                candidates, var_name,
                                &[OwnerQual::Stack, OwnerQual::Owned, OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask],
                                alias_of,
                            );
                            // A def call on a non-mut auto-ref param is a constraint signal
                            // (prevents auto-ref, since &Counter can't support def calls).
                            // For mut params, def calls are expected — auto-ref still applies.
                            if is_auto_ref_param && !is_mut_param {
                                has_qualifier_constraint.insert(var_name.clone());
                            }
                        }
                    }
                }
            }
            ExprKind::BinOp(_, l, r) => {
                self.walk_expr_for_qualifiers(l, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                self.walk_expr_for_qualifiers(r, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
            }
            ExprKind::UnaryOp(_, e) => {
                self.walk_expr_for_qualifiers(e, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
            }
            ExprKind::If(if_stmt) => {
                for (cond, body) in &if_stmt.branches {
                    self.walk_expr_for_qualifiers(cond, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                    for st in body {
                        self.walk_stmt_for_qualifiers(st, anonymous_vars, var_struct_types, alias_of, None, candidates, auto_ref_param_vars, has_qualifier_constraint);
                    }
                }
                if let Some(else_body) = &if_stmt.else_body {
                    for st in else_body {
                        self.walk_stmt_for_qualifiers(st, anonymous_vars, var_struct_types, alias_of, None, candidates, auto_ref_param_vars, has_qualifier_constraint);
                    }
                }
            }
            // Task capture: variables captured in task bodies need Arc-based qualifiers.
            // Receiver of method call → needs mutation → {Actor, Guard}.
            // Non-receiver → read-only → {Shared, Actor, Guard}.
            ExprKind::Task(inner) => {
                self.constrain_task_captures(inner, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                self.walk_expr_for_qualifiers(inner, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
            }
            ExprKind::TaskWithTimeout(dur, inner) => {
                self.walk_expr_for_qualifiers(dur, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                self.constrain_task_captures(inner, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                self.walk_expr_for_qualifiers(inner, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
            }
            // Assignment is a mutation signal: eliminates Shared — but only for mutation
            // *through* the value (`x.field = val`, `x[i] = val`, `x.a.b.c = val`), which
            // requires the pointee to support interior mutability. A plain whole-value
            // rebind (`x = val`) just replaces the local binding itself and is legal for
            // `'shared` (Rc/Arc) too, so it must not eliminate it.
            ExprKind::Assign(target, val) => {
                if !matches!(target.kind, ExprKind::Var(_)) {
                    if let Some(var_name) = mutation_root(target) {
                        if anonymous_vars.contains(var_name) {
                            constrain_candidates(
                                candidates, var_name,
                                &[OwnerQual::Stack, OwnerQual::Owned, OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask],
                                alias_of,
                            );
                            if auto_ref_param_vars.contains(var_name) {
                                has_qualifier_constraint.insert(var_name.to_string());
                            }
                        }
                    }
                }
                // Field assignment with 'actor RHS: `param.field = actor_val`
                // → tighten the owning param's candidates to {Actor, Guard}.
                if let ExprKind::Field(obj, _) = &target.kind {
                    if let ExprKind::Var(v) = &obj.kind {
                        if anonymous_vars.contains(v.as_str()) && self.expr_returns_actor_qual(val) {
                            constrain_candidates(
                                candidates, v.as_str(),
                                &[OwnerQual::Actor, OwnerQual::Guard],
                                alias_of,
                            );
                            has_qualifier_constraint.insert(v.to_string());
                        }
                    }
                }
                self.walk_expr_for_qualifiers(target, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                self.walk_expr_for_qualifiers(val, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
            }
            // Closure captures: same logic as task captures.
            // A closure that captures x as a method receiver needs mutation → {Actor, Guard}.
            // A closure that only reads x → {Shared, Actor, Guard}.
            ExprKind::Closure(_, _, body, _, _) => {
                use crate::ast::ClosureBody;
                match body {
                    ClosureBody::Expr(e) => {
                        self.constrain_task_captures(e, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                        self.walk_expr_for_qualifiers(e, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                    }
                    ClosureBody::Block(stmts) => {
                        let block_expr = Expr {
                            kind: ExprKind::Block(stmts.clone()),
                            line: 0, col: 0, len: 0,
                        };
                        self.constrain_task_captures(&block_expr, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                        for st in stmts {
                            self.walk_stmt_for_qualifiers(st, anonymous_vars, var_struct_types, alias_of, None, candidates, auto_ref_param_vars, has_qualifier_constraint);
                        }
                    }
                }
            }
            // Match expression: a param used as match subject must be taken by value.
            ExprKind::Match(s) => {
                if let ExprKind::Var(vname) = &s.subject.kind {
                    if auto_ref_param_vars.contains(vname.as_str()) {
                        has_qualifier_constraint.insert(vname.clone());
                    }
                }
                self.walk_expr_for_qualifiers(&s.subject, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                for arm in &s.arms {
                    if let Some(guard) = &arm.guard {
                        self.walk_expr_for_qualifiers(guard, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                    }
                    match &arm.body {
                        crate::ast::MatchBody::Expr(e) => {
                            self.walk_expr_for_qualifiers(e, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                        }
                        crate::ast::MatchBody::Block(stmts) => {
                            for st in stmts {
                                self.walk_stmt_for_qualifiers(st, anonymous_vars, var_struct_types, alias_of, None, candidates, auto_ref_param_vars, has_qualifier_constraint);
                            }
                        }
                    }
                }
            }
            ExprKind::New { arena, ctor } => {
                if let Some(arena_expr) = arena {
                    self.walk_expr_for_qualifiers(arena_expr, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                }
                self.walk_expr_for_qualifiers(ctor, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
            }
            _ => {}
        }
    }

    /// Walk a CondClause for qualifier constraints.
    /// Moving-out patterns (`let Some(x) = param.field`) suppress auto-ref on the root param.
    #[allow(clippy::too_many_arguments)]
    fn walk_cond_clause_for_qualifiers(
        &mut self,
        clause: &crate::ast::CondClause,
        anonymous_vars: &std::collections::HashSet<String>,
        var_struct_types: &std::collections::HashMap<String, String>,
        alias_of: &std::collections::HashMap<String, String>,
        candidates: &mut std::collections::HashMap<String, Vec<OwnerQual>>,
        auto_ref_param_vars: &std::collections::HashSet<String>,
        has_qualifier_constraint: &mut std::collections::HashSet<String>,
    ) {
        let expr = match clause {
            crate::ast::CondClause::Let(_, e) | crate::ast::CondClause::LetPat(_, e) => e,
            crate::ast::CondClause::Expr(e) => {
                self.walk_expr_for_qualifiers(e, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
                return;
            }
        };
        // Suppress auto-ref if the source is a field of an auto-ref param (moving-out pattern).
        if let ExprKind::Field(obj, _) = &expr.kind {
            if let ExprKind::Var(vname) = &obj.kind {
                if auto_ref_param_vars.contains(vname.as_str()) {
                    has_qualifier_constraint.insert(vname.clone());
                }
            }
        }
        self.walk_expr_for_qualifiers(expr, anonymous_vars, var_struct_types, alias_of, candidates, auto_ref_param_vars, has_qualifier_constraint);
    }

    /// Constrain qualifiers for variables captured by a task body.
    /// All captured vars must be Arc-based (crossable across async boundaries).
    /// Receivers of method calls additionally need mutation → {Actor, Guard}.
    /// Both the plain (`Actor`/`Guard`) and `'task` (`ActorTask`/`GuardTask`) variants are
    /// kept as candidates here — the final pick between them happens in `infer_qualifiers`'s
    /// resolution loop via `disambiguate_task_variant`, based on whether a `task`-declared
    /// method was called on the captured variable (recorded here into `task_method_call_vars`).
    #[allow(clippy::too_many_arguments)]
    fn constrain_task_captures(
        &mut self,
        body: &Expr,
        anonymous_vars: &std::collections::HashSet<String>,
        var_struct_types: &std::collections::HashMap<String, String>,
        alias_of: &std::collections::HashMap<String, String>,
        candidates: &mut std::collections::HashMap<String, Vec<OwnerQual>>,
        auto_ref_param_vars: &std::collections::HashSet<String>,
        has_qualifier_constraint: &mut std::collections::HashSet<String>,
    ) {
        let captured: std::collections::HashSet<String> = collect_var_names(body).into_iter().collect();
        let receiver_methods = method_receivers(body);

        for var_name in &captured {
            if !anonymous_vars.contains(var_name.as_str()) { continue; }
            // Task capture is a storage signal: marks the param as non-auto-ref.
            if auto_ref_param_vars.contains(var_name.as_str()) {
                has_qualifier_constraint.insert(var_name.clone());
            }
            let called_methods = receiver_methods.get(var_name.as_str());
            promote_task_variants(candidates, var_name, alias_of);
            let compatible: &[OwnerQual] = if let Some(methods) = called_methods {
                if let Some(struct_name) = var_struct_types.get(var_name.as_str()) {
                    let is_task_call = methods.iter().any(|m| {
                        self.struct_task_methods.contains(&format!("{}::{}", struct_name, m))
                    });
                    if is_task_call {
                        self.task_method_call_vars.insert(var_name.clone());
                    }
                }
                &[OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask]
            } else {
                &[OwnerQual::Shared, OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask]
            };
            constrain_candidates(candidates, var_name, compatible, alias_of);
        }
    }

    /// Infer qualifiers for private, unqualified struct fields by scanning all method bodies
    /// in the same struct, plus any `ext` block methods/setters for the same type declared in
    /// the same file. The same constraint-elimination algorithm used for local variables is
    /// applied to `self.field` accesses across all of them.
    ///
    /// Results are written directly into `struct_mutex_fields` and `struct_rwlock_fields` so
    /// the existing field-access emission infrastructure handles wrapping/unwrapping automatically.
    /// Only private fields (`is_pub == false`) with no explicit qualifier are considered.
    ///
    /// `ext` blocks in *other* files are not visible here — cross-file inference is out of scope
    /// (see docs/qualifiers.md).
    pub(crate) fn infer_struct_field_qualifiers(
        &mut self,
        s: &crate::ast::StructDecl,
        ext_methods: &[&crate::ast::FnDecl],
        ext_setters: &[&crate::ast::SetDecl],
    ) {

        // Collect private, unqualified fields with their declared inner type name.
        let target_fields: std::collections::HashMap<String, String> = s.fields.iter()
            // `mut Point p` wraps `f.ty` in `Type::Mut` (docs/mut-type-modifier.md
            // §3) — strip it before inspecting the shape, same as everywhere else.
            .filter(|f| !matches!(f.ty.without_mut(), Type::Qualified(..)))
            .filter_map(|f| {
                // Only struct-typed fields (Named type) are candidates for qualifier inference.
                if let Type::Named(type_name) = f.ty.without_mut() {
                    Some((f.name.clone(), type_name.clone()))
                } else {
                    None
                }
            })
            .collect();

        if target_fields.is_empty() { return; }

        self.task_method_call_fields.clear();

        let mut candidates: std::collections::HashMap<String, Vec<OwnerQual>> = target_fields
            .keys()
            .map(|name| (name.clone(), all_qualifiers()))
            .collect();

        let empty_alias: std::collections::HashMap<String, String> = std::collections::HashMap::new();

        // Walk every method body looking for self.field access patterns.
        for method in &s.methods {
            self.walk_stmts_for_field_qualifiers(
                &method.body,
                &s.name,
                &target_fields,
                &empty_alias,
                &mut candidates,
            );
        }
        for setter in &s.setters {
            self.walk_stmts_for_field_qualifiers(
                &setter.body,
                &s.name,
                &target_fields,
                &empty_alias,
                &mut candidates,
            );
        }
        for method in ext_methods {
            self.walk_stmts_for_field_qualifiers(
                &method.body,
                &s.name,
                &target_fields,
                &empty_alias,
                &mut candidates,
            );
        }
        for setter in ext_setters {
            self.walk_stmts_for_field_qualifiers(
                &setter.body,
                &s.name,
                &target_fields,
                &empty_alias,
                &mut candidates,
            );
        }

        // Resolve candidates for struct fields.
        for (field_name, remaining) in &candidates {
            let key = format!("{}::{}", s.name, field_name);
            // If both a plain qualifier and its 'task variant survived elimination, pick
            // one based on whether a task-declared method was called on this field.
            let mut remaining: Vec<OwnerQual> = remaining.clone();
            disambiguate_task_variant(&mut remaining, self.task_method_call_fields.contains(field_name.as_str()));
            let resolved = match remaining.len() {
                0 => continue,
                1 => remaining[0].clone(),
                _ => {
                    // Multi-candidate fallback for struct fields.
                    // Struct fields are always laid out inline in the parent allocation,
                    // so 'stack is always preferred when available.
                    let type_size = target_fields.get(field_name.as_str())
                        .and_then(|tn| self.type_sizes.get(tn.as_str()))
                        .copied();
                    match resolve_fallback(&remaining, true, type_size, self.config.stack_auto_bytes) {
                        Some(q) => q,
                        None => continue,
                    }
                }
            };
            match &resolved {
                OwnerQual::Actor    => { self.struct_mutex_fields.insert(key); }
                OwnerQual::ActorTask => { self.struct_mutex_task_fields.insert(key); }
                OwnerQual::Guard    => { self.struct_rwlock_fields.insert(key); }
                OwnerQual::GuardTask => { self.struct_rwlock_task_fields.insert(key); }
                // Stack / Owned / Shared: no registry needed — plain T or Box<T>.
                _ => {}
            }
        }
    }

    fn walk_stmts_for_field_qualifiers(
        &mut self,
        stmts: &[Stmt],
        struct_name: &str,
        target_fields: &std::collections::HashMap<String, String>,
        alias_of: &std::collections::HashMap<String, String>,
        candidates: &mut std::collections::HashMap<String, Vec<OwnerQual>>,
    ) {
        for stmt in stmts {
            self.walk_stmt_for_field_qualifiers(stmt, struct_name, target_fields, alias_of, candidates);
        }
    }

    fn walk_stmt_for_field_qualifiers(
        &mut self,
        stmt: &Stmt,
        struct_name: &str,
        target_fields: &std::collections::HashMap<String, String>,
        alias_of: &std::collections::HashMap<String, String>,
        candidates: &mut std::collections::HashMap<String, Vec<OwnerQual>>,
    ) {
        match stmt {
            Stmt::Let(s) => {
                if let Some(val) = &s.value {
                    self.walk_expr_for_field_qualifiers(val, struct_name, target_fields, alias_of, candidates);
                }
            }
            Stmt::Expr(e) | Stmt::Return(crate::ast::ReturnStmt { value: Some(e), .. }) => {
                self.walk_expr_for_field_qualifiers(e, struct_name, target_fields, alias_of, candidates);
            }
            Stmt::If(s) => {
                for (cond, body) in &s.branches {
                    self.walk_expr_for_field_qualifiers(cond, struct_name, target_fields, alias_of, candidates);
                    self.walk_stmts_for_field_qualifiers(body, struct_name, target_fields, alias_of, candidates);
                }
                if let Some(eb) = &s.else_body {
                    self.walk_stmts_for_field_qualifiers(eb, struct_name, target_fields, alias_of, candidates);
                }
            }
            Stmt::While(s) => {
                self.walk_expr_for_field_qualifiers(&s.condition, struct_name, target_fields, alias_of, candidates);
                self.walk_stmts_for_field_qualifiers(&s.body, struct_name, target_fields, alias_of, candidates);
            }
            Stmt::For(s) => {
                self.walk_expr_for_field_qualifiers(&s.iterable, struct_name, target_fields, alias_of, candidates);
                self.walk_stmts_for_field_qualifiers(&s.body, struct_name, target_fields, alias_of, candidates);
            }
            Stmt::Match(s) => {
                self.walk_expr_for_field_qualifiers(&s.subject, struct_name, target_fields, alias_of, candidates);
                for arm in &s.arms {
                    match &arm.body {
                        MatchBody::Expr(e) => {
                            self.walk_expr_for_field_qualifiers(e, struct_name, target_fields, alias_of, candidates);
                        }
                        MatchBody::Block(stmts) => {
                            self.walk_stmts_for_field_qualifiers(stmts, struct_name, target_fields, alias_of, candidates);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn walk_expr_for_field_qualifiers(
        &mut self,
        expr: &Expr,
        struct_name: &str,
        target_fields: &std::collections::HashMap<String, String>,
        alias_of: &std::collections::HashMap<String, String>,
        candidates: &mut std::collections::HashMap<String, Vec<OwnerQual>>,
    ) {
        match &expr.kind {
            ExprKind::Call(callee, args) => {
                self.walk_expr_for_field_qualifiers(callee, struct_name, target_fields, alias_of, candidates);
                for arg in args {
                    self.walk_expr_for_field_qualifiers(&arg.value, struct_name, target_fields, alias_of, candidates);
                }
                // self.field passed to a function demanding a concrete qualifier.
                if let ExprKind::Var(fn_name) = &callee.kind {
                    let param_types = self.fn_sigs.get(fn_name.as_str()).cloned();
                    if let Some(param_types) = param_types {
                        for (i, arg) in args.iter().enumerate() {
                            let Some(demanded) = param_types.get(i).and_then(qual_of_type) else { continue };
                            let Some(field_name) = self_field_name(&arg.value) else { continue };
                            if target_fields.contains_key(field_name) {
                                constrain_candidates(candidates, field_name, &coercible_from(demanded), alias_of);
                            }
                        }
                    }
                }
            }
            ExprKind::MethodCall(obj, method, args) => {
                for arg in args {
                    self.walk_expr_for_field_qualifiers(&arg.value, struct_name, target_fields, alias_of, candidates);
                }
                // self.field.method() — check if it's a def (mutating) call.
                if let Some(field_name) = self_field_name(obj) {
                    if let Some(field_struct_type) = target_fields.get(field_name) {
                        let is_req = self.struct_req_methods
                            .contains(&format!("{}::{}", field_struct_type, method));
                        if !is_req {
                            constrain_candidates(
                                candidates, field_name,
                                &[OwnerQual::Stack, OwnerQual::Owned, OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask],
                                alias_of,
                            );
                        }
                    }
                } else {
                    self.walk_expr_for_field_qualifiers(obj, struct_name, target_fields, alias_of, candidates);
                }
            }
            ExprKind::BinOp(_, l, r) => {
                self.walk_expr_for_field_qualifiers(l, struct_name, target_fields, alias_of, candidates);
                self.walk_expr_for_field_qualifiers(r, struct_name, target_fields, alias_of, candidates);
            }
            ExprKind::UnaryOp(_, e) => {
                self.walk_expr_for_field_qualifiers(e, struct_name, target_fields, alias_of, candidates);
            }
            ExprKind::If(if_stmt) => {
                for (cond, body) in &if_stmt.branches {
                    self.walk_expr_for_field_qualifiers(cond, struct_name, target_fields, alias_of, candidates);
                    self.walk_stmts_for_field_qualifiers(body, struct_name, target_fields, alias_of, candidates);
                }
                if let Some(eb) = &if_stmt.else_body {
                    self.walk_stmts_for_field_qualifiers(eb, struct_name, target_fields, alias_of, candidates);
                }
            }
            // Task capture: self.field captured in a task body.
            ExprKind::Task(inner) | ExprKind::TaskWithTimeout(_, inner) => {
                self.constrain_task_field_captures(inner, target_fields, alias_of, candidates);
                self.walk_expr_for_field_qualifiers(inner, struct_name, target_fields, alias_of, candidates);
            }
            _ => {}
        }
    }

    /// Constrain qualifiers for struct fields captured by a task body.
    /// Mirrors `constrain_task_captures`: keeps both the plain and `'task` variant as
    /// candidates, recording a task-method-call signal into `task_method_call_fields` for
    /// the later `disambiguate_task_variant` tie-break.
    fn constrain_task_field_captures(
        &mut self,
        body: &Expr,
        target_fields: &std::collections::HashMap<String, String>,
        alias_of: &std::collections::HashMap<String, String>,
        candidates: &mut std::collections::HashMap<String, Vec<OwnerQual>>,
    ) {
        let receiver_methods = method_receivers(body);
        let accessed = self_field_names_in_expr(body);

        for field_name in &accessed {
            if !target_fields.contains_key(field_name.as_str()) { continue; }
            let called_methods = receiver_methods.get(field_name.as_str());
            promote_task_variants(candidates, field_name, alias_of);
            let compatible: &[OwnerQual] = if let Some(methods) = called_methods {
                if let Some(struct_name) = target_fields.get(field_name.as_str()) {
                    let is_task_call = methods.iter().any(|m| {
                        self.struct_task_methods.contains(&format!("{}::{}", struct_name, m))
                    });
                    if is_task_call {
                        self.task_method_call_fields.insert(field_name.clone());
                    }
                }
                &[OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask]
            } else {
                &[OwnerQual::Shared, OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask]
            };
            constrain_candidates(candidates, field_name, compatible, alias_of);
        }
    }

    /// Post-inference pass: for every call site where a parameter type is a qualifier union,
    /// check that the argument's qualifier (inferred or explicitly declared) is a member of
    /// the allowed set. Emits an error if a disallowed qualifier is found.
    pub(crate) fn validate_union_constraints(&self, stmts: &[Stmt]) {
        for stmt in stmts {
            self.validate_stmt(stmt);
        }
    }

    fn validate_stmt(&self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(s) => {
                if let Some(val) = &s.value { self.validate_expr(val); }
            }
            Stmt::Expr(e) | Stmt::Return(crate::ast::ReturnStmt { value: Some(e), .. }) => {
                self.validate_expr(e);
            }
            Stmt::If(s) => {
                for (cond, body) in &s.branches {
                    self.validate_expr(cond);
                    for st in body { self.validate_stmt(st); }
                }
                if let Some(eb) = &s.else_body {
                    for st in eb { self.validate_stmt(st); }
                }
            }
            Stmt::While(s) => {
                self.validate_expr(&s.condition);
                for st in &s.body { self.validate_stmt(st); }
            }
            Stmt::For(s) => {
                self.validate_expr(&s.iterable);
                for st in &s.body { self.validate_stmt(st); }
            }
            Stmt::Match(s) => {
                self.validate_expr(&s.subject);
                for arm in &s.arms {
                    match &arm.body {
                        MatchBody::Expr(e) => self.validate_expr(e),
                        MatchBody::Block(stmts) => {
                            for st in stmts { self.validate_stmt(st); }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn validate_expr(&self, expr: &Expr) {
        match &expr.kind {
            ExprKind::Call(callee, args) => {
                for arg in args { self.validate_expr(&arg.value); }
                if let ExprKind::Var(fn_name) = &callee.kind {
                    let param_types = self.fn_sigs.get(fn_name.as_str()).cloned();
                    if let Some(param_types) = param_types {
                        for (i, arg) in args.iter().enumerate() {
                            let Some(param_ty) = param_types.get(i) else { continue };
                            let ExprKind::Var(var_name) = &arg.value.kind else { continue };

                            // Caller check: Union-typed parameter → argument qualifier must be in the union.
                            if let Type::Qualified(_, OwnerQual::Union(members)) = param_ty {
                                let Some(arg_qual) = self.var_qual(var_name) else { continue };
                                if !members.iter().any(|m| quals_equal(m, &arg_qual)) {
                                    let allowed: Vec<&str> = members.iter().map(|q| qual_name(q)).collect();
                                    eprintln!(
                                        "error line {}: qualifier '{}' for `{}` is not allowed here\n  \
                                         → parameter {} of `{}` accepts only: {}\n  \
                                         fix: annotate `{}` with one of the listed qualifiers",
                                        expr.line, qual_name(&arg_qual), var_name,
                                        i + 1, fn_name, allowed.join("|"),
                                        var_name
                                    );
                                }
                            }

                            // Body-compatibility check: Union-typed argument passed where a concrete
                            // qualifier is demanded — verify the demanded qualifier is in the union.
                            if let Some(demanded) = qual_of_type(param_ty) {
                                if let Some(arg_union) = self.var_union(var_name) {
                                    if !arg_union.iter().any(|m| quals_equal(m, &demanded)) {
                                        let union_s: Vec<&str> = arg_union.iter().map(|q| qual_name(q)).collect();
                                        eprintln!(
                                            "error line {}: `{}` has qualifier constraint '{}'\n  \
                                             → this call demands '{}' which is outside the constraint\n  \
                                             fix: change the qualifier constraint on `{}` to include '{}', \
                                             or pick a concrete qualifier",
                                            expr.line, var_name, union_s.join("|"),
                                            qual_name(&demanded), var_name, qual_name(&demanded)
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
            ExprKind::MethodCall(obj, _, args) => {
                self.validate_expr(obj);
                for arg in args { self.validate_expr(&arg.value); }
            }
            ExprKind::BinOp(_, l, r) => { self.validate_expr(l); self.validate_expr(r); }
            ExprKind::UnaryOp(_, e) => { self.validate_expr(e); }
            ExprKind::If(if_stmt) => {
                for (cond, body) in &if_stmt.branches {
                    self.validate_expr(cond);
                    for st in body { self.validate_stmt(st); }
                }
                if let Some(eb) = &if_stmt.else_body {
                    for st in eb { self.validate_stmt(st); }
                }
            }
            _ => {}
        }
    }

    /// Get the resolved qualifier for a named variable:
    /// first checks inferred qualifiers, then the variable's declared type.
    fn var_qual(&self, name: &str) -> Option<OwnerQual> {
        if let Some(q) = self.inferred_qualifiers.get(name) {
            return Some(q.clone());
        }
        if let Some(ty) = self.var_types.get(name) {
            return qual_of_type(ty);
        }
        None
    }

    /// If the variable has a Union qualifier (declared), return the member list.
    fn var_union<'a>(&'a self, name: &str) -> Option<&'a Vec<OwnerQual>> {
        let ty = self.fn_current_params.get(name).or_else(|| self.var_types.get(name))?;
        if let Type::Qualified(_, OwnerQual::Union(members)) = ty.without_mut() {
            return Some(members);
        }
        None
    }

    /// After inference: for each unqualified parameter whose body uses demanded a concrete
    /// qualifier, emit a hint suggesting an explicit annotation on the parameter.
    pub(crate) fn suggest_param_annotations(&self) {
        for (param_name, param_ty) in &self.fn_current_params {
            if qual_of_type(param_ty).is_some() { continue; }
            if matches!(param_ty, Type::Qualified(_, OwnerQual::Union(_))) { continue; }
            if let Some(inferred) = self.inferred_qualifiers.get(param_name.as_str()) {
                // Auto-ref (Borrow / BorrowMut) is resolved silently — no annotation needed.
                if matches!(inferred, OwnerQual::Borrow | OwnerQual::BorrowMut) { continue; }
                let line = self.fn_current_param_lines.get(param_name.as_str()).copied().unwrap_or(0);
                eprintln!(
                    "hint: parameter `{}` is always used as '{}' in this body\n  \
                     → consider annotating it explicitly to make the contract clear at call sites\n \
                     --> line {}",
                    param_name, qual_name(inferred), line
                );
            }
        }
    }

    /// Returns true if `expr` evaluates to a value whose qualifier is 'actor or 'guard.
    /// Handles: calls to functions with declared 'actor/'guard return types, and variables
    /// already known to be 'actor (via var_mutex_types, var_mutex_task_types, or
    /// infer_local_actor_vars populated by the pre-pass in infer_qualifiers).
    pub(crate) fn expr_returns_actor_qual(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Call(callee, _) => {
                if let ExprKind::Var(fn_name) = &callee.kind {
                    matches!(
                        self.fn_return_types.get(fn_name.as_str()),
                        Some(Type::Qualified(_, OwnerQual::Actor | OwnerQual::Guard))
                    )
                } else {
                    false
                }
            }
            ExprKind::Var(vname) => {
                self.infer_local_actor_vars.contains(vname.as_str())
                    || self.var_mutex_types.contains(vname.as_str())
                    || self.var_mutex_task_types.contains(vname.as_str())
            }
            _ => false,
        }
    }
}

/// Intersect the candidate set for `var_name` (and its aliases) with `compatible`.
/// Qualifiers not in `compatible` are eliminated.
fn constrain_candidates(
    candidates: &mut std::collections::HashMap<String, Vec<OwnerQual>>,
    var_name: &str,
    compatible: &[OwnerQual],
    alias_of: &std::collections::HashMap<String, String>,
) {
    let root = alias_of.get(var_name).map(|s| s.as_str()).unwrap_or(var_name).to_string();

    if let Some(list) = candidates.get_mut(&root) {
        list.retain(|q| compatible.iter().any(|c| quals_equal(q, c)));
    }
    for (alias, target) in alias_of {
        if target.as_str() == root.as_str() {
            if let Some(list) = candidates.get_mut(alias) {
                list.retain(|q| compatible.iter().any(|c| quals_equal(q, c)));
            }
        }
    }
}

/// Add `ActorTask`/`GuardTask` to a variable's (and its aliases') candidate set wherever
/// the corresponding plain variant (`Actor`/`Guard`) is already a candidate. Called when a
/// task/closure capture is detected — both variants remain viable until
/// `disambiguate_task_variant` picks one at resolution time.
fn promote_task_variants(
    candidates: &mut std::collections::HashMap<String, Vec<OwnerQual>>,
    var_name: &str,
    alias_of: &std::collections::HashMap<String, String>,
) {
    let root = alias_of.get(var_name).map(|s| s.as_str()).unwrap_or(var_name).to_string();
    let mut keys: Vec<String> = vec![root.clone()];
    for (alias, target) in alias_of {
        if target.as_str() == root.as_str() {
            keys.push(alias.clone());
        }
    }
    for key in &keys {
        if let Some(list) = candidates.get_mut(key.as_str()) {
            if list.iter().any(|q| quals_equal(q, &OwnerQual::Actor))
                && !list.iter().any(|q| quals_equal(q, &OwnerQual::ActorTask))
            {
                list.push(OwnerQual::ActorTask);
            }
            if list.iter().any(|q| quals_equal(q, &OwnerQual::Guard))
                && !list.iter().any(|q| quals_equal(q, &OwnerQual::GuardTask))
            {
                list.push(OwnerQual::GuardTask);
            }
        }
    }
}

/// When both a plain qualifier (`Actor`/`Guard`) and its `'task` variant remain as
/// candidates after constraint elimination, pick one based on whether a `task`-declared
/// method was called on the variable (`has_task_call`, from `task_method_call_vars` /
/// `task_method_call_fields`). This is the tie-break the doc's open question on
/// `'actor'task` vs `'actor` inference asked for.
fn disambiguate_task_variant(remaining: &mut Vec<OwnerQual>, has_task_call: bool) {
    let has_actor = remaining.iter().any(|q| quals_equal(q, &OwnerQual::Actor));
    let has_actor_task = remaining.iter().any(|q| quals_equal(q, &OwnerQual::ActorTask));
    if has_actor && has_actor_task {
        if has_task_call {
            remaining.retain(|q| !quals_equal(q, &OwnerQual::Actor));
        } else {
            remaining.retain(|q| !quals_equal(q, &OwnerQual::ActorTask));
        }
    }
    let has_guard = remaining.iter().any(|q| quals_equal(q, &OwnerQual::Guard));
    let has_guard_task = remaining.iter().any(|q| quals_equal(q, &OwnerQual::GuardTask));
    if has_guard && has_guard_task {
        if has_task_call {
            remaining.retain(|q| !quals_equal(q, &OwnerQual::Guard));
        } else {
            remaining.retain(|q| !quals_equal(q, &OwnerQual::GuardTask));
        }
    }
}

/// For 'shared/'actor/'guard demands, 'stack and 'heap are also acceptable
/// because a plain T or Box<T> can be wrapped at the call site.
/// For 'stack/'heap demands, only the exact qualifier is accepted.
fn coercible_from(demanded: OwnerQual) -> Vec<OwnerQual> {
    match demanded {
        OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard =>
            vec![OwnerQual::Stack, OwnerQual::Owned, demanded],
        // Universal immutable borrow: any qualifier is accepted — no constraint on caller.
        OwnerQual::Borrow => all_qualifiers(),
        // Universal mutable borrow: any mutable qualifier ('shared excluded).
        OwnerQual::BorrowMut =>
            vec![OwnerQual::Stack, OwnerQual::Owned, OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask],
        _ => vec![demanded],
    }
}

fn all_qualifiers() -> Vec<OwnerQual> {
    vec![
        OwnerQual::Stack,
        OwnerQual::Owned,
        OwnerQual::Shared,
        OwnerQual::Actor,
        OwnerQual::Guard,
    ]
}

/// Priority-ordered fallback when multiple qualifier candidates remain after constraint
/// elimination.
///
/// 1. If `Stack` ∈ candidates:
///    - struct field (any binding) → `'stack` (bytes are part of parent allocation)
///    - local variable, sizeof(T) ≤ stack_auto_bytes → `'stack`
///    - type too large → skip `'stack`, go to ordered chain
///
/// 2. Ordered chain: `'heap` > `'shared` > `'actor`(/`'actor'task`) > `'guard`(/`'guard'task`)
fn resolve_fallback(
    candidates: &[OwnerQual],
    is_struct_field: bool,
    type_size: Option<usize>,
    stack_auto_bytes: usize,
) -> Option<OwnerQual> {
    let has = |q: &OwnerQual| candidates.iter().any(|c| quals_equal(c, q));
    let fits = type_size.is_none_or(|s| s <= stack_auto_bytes);

    // Ordered chain: 'heap > 'shared > 'actor(/'actor'task) > 'guard(/'guard'task).
    // The 'task variant is checked first at each slot so that it wins when it's the one
    // that survived constraint elimination (e.g. after `disambiguate_task_variant`) —
    // by that point at most one of {Actor, ActorTask} and one of {Guard, GuardTask} remain.
    fn tail_pick(has: &dyn Fn(&OwnerQual) -> bool) -> Option<OwnerQual> {
        if has(&OwnerQual::Owned) { return Some(OwnerQual::Owned); }
        if has(&OwnerQual::Shared) { return Some(OwnerQual::Shared); }
        if has(&OwnerQual::ActorTask) { return Some(OwnerQual::ActorTask); }
        if has(&OwnerQual::Actor) { return Some(OwnerQual::Actor); }
        if has(&OwnerQual::GuardTask) { return Some(OwnerQual::GuardTask); }
        if has(&OwnerQual::Guard) { return Some(OwnerQual::Guard); }
        None
    }

    // Step 1: 'stack — struct field of any binding, or small local variable.
    if has(&OwnerQual::Stack) {
        if is_struct_field {
            return Some(OwnerQual::Stack);
        }
        if fits {
            return Some(OwnerQual::Stack);
        }
        return tail_pick(&has);
    }

    // Step 2: neither 'stack nor other stack-allocated in candidates — first from the ordered chain.
    tail_pick(&has)
}

/// Candidate set for T' (tick) variables: indirection is certain, kind is inferred.
fn indirection_qualifiers() -> Vec<OwnerQual> {
    vec![
        OwnerQual::Owned,
        OwnerQual::Shared,
        OwnerQual::Actor,
        OwnerQual::Guard,
    ]
}

fn collect_anonymous_vars(
    stmt: &Stmt,
    anonymous_vars: &mut std::collections::HashSet<String>,
    alias_of: &mut std::collections::HashMap<String, String>,
    var_struct_types: &mut std::collections::HashMap<String, String>,
    mut_bindings: &mut std::collections::HashSet<String>,
    tick_bindings: &mut std::collections::HashSet<String>,
) {
    match stmt {
        Stmt::Let(s) => {
            let is_tick = match &s.ty {
                Some(Type::Qualified(_, OwnerQual::Owned | OwnerQual::New)) => true,
                Some(Type::Optional(inner)) => matches!(inner.as_ref(), Type::Qualified(_, OwnerQual::Owned | OwnerQual::New)),
                _ => false,
            };
            let is_anonymous = is_tick || match &s.ty {
                None => true,
                Some(Type::Named(_)) => true,
                Some(Type::Optional(inner)) => matches!(inner.as_ref(), Type::Named(_)),
                _ => false,
            };
            if is_anonymous {
                anonymous_vars.insert(s.name.clone());
                // T' or T'? binding: indirection hint, restricts to {Owned, Shared, Actor, Guard}.
                if is_tick {
                    tick_bindings.insert(s.name.clone());
                }
                // `mut` binding: mutation signal at declaration site.
                if s.binding == BindingKind::Mut {
                    mut_bindings.insert(s.name.clone());
                }
                if let Some(val) = &s.value {
                    // `new Constructor()` on RHS without arena: treat as tick binding
                    // (infer excluding 'stack), same as T' bare tick.
                    if !is_tick {
                        if let ExprKind::New { arena: None, ctor } = &val.kind {
                            tick_bindings.insert(s.name.clone());
                            // Also populate var_struct_types from the ctor callee.
                            if let ExprKind::Call(callee, _) = &ctor.kind {
                                if let ExprKind::Var(type_name) = &callee.kind {
                                    if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                                        var_struct_types.insert(s.name.clone(), type_name.clone());
                                    }
                                }
                            }
                        }
                    }
                    match &val.kind {
                        ExprKind::Var(src) => {
                            alias_of.insert(s.name.clone(), src.clone());
                        }
                        // some(Counter(0)) — capture inner struct type for optional vars; must come before generic Call arm
                        ExprKind::Call(callee, args)
                            if matches!(&callee.kind, ExprKind::Var(n) if n.as_str() == "some") =>
                        {
                            if let Some(arg) = args.first() {
                                if let ExprKind::Call(inner_callee, _) = &arg.value.kind {
                                    if let ExprKind::Var(type_name) = &inner_callee.kind {
                                        if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                                            var_struct_types.insert(s.name.clone(), type_name.clone());
                                        }
                                    }
                                }
                            }
                        }
                        ExprKind::Call(callee, _) => {
                            if let ExprKind::Var(type_name) = &callee.kind {
                                if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                                    var_struct_types.insert(s.name.clone(), type_name.clone());
                                }
                            }
                        }
                        // `new(arena) Constructor()` — populate var_struct_types from ctor callee.
                        ExprKind::New { ctor, .. } => {
                            if let ExprKind::Call(callee, _) = &ctor.kind {
                                if let ExprKind::Var(type_name) = &callee.kind {
                                    if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                                        var_struct_types.insert(s.name.clone(), type_name.clone());
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Stmt::If(s) => {
            for (_, body) in &s.branches {
                for st in body { collect_anonymous_vars(st, anonymous_vars, alias_of, var_struct_types, mut_bindings, tick_bindings); }
            }
            if let Some(else_body) = &s.else_body {
                for st in else_body { collect_anonymous_vars(st, anonymous_vars, alias_of, var_struct_types, mut_bindings, tick_bindings); }
            }
        }
        Stmt::While(s) => {
            for st in &s.body { collect_anonymous_vars(st, anonymous_vars, alias_of, var_struct_types, mut_bindings, tick_bindings); }
        }
        Stmt::For(s) => {
            for st in &s.body { collect_anonymous_vars(st, anonymous_vars, alias_of, var_struct_types, mut_bindings, tick_bindings); }
        }
        Stmt::Match(s) => {
            for arm in &s.arms {
                match &arm.body {
                    MatchBody::Block(stmts) => {
                        for st in stmts { collect_anonymous_vars(st, anonymous_vars, alias_of, var_struct_types, mut_bindings, tick_bindings); }
                    }
                    MatchBody::Expr(_) => {}
                }
            }
        }
        _ => {}
    }
}

/// Walk an assignment target expression to find the root variable name.
/// Handles arbitrary nesting: `x`, `x.field`, `x[i]`, `x.a.b[i].c`, etc.
fn mutation_root(expr: &Expr) -> Option<&str> {
    match &expr.kind {
        ExprKind::Var(n) => Some(n.as_str()),
        ExprKind::Field(obj, _) | ExprKind::Index(obj, _) => mutation_root(obj),
        _ => None,
    }
}

fn qual_of_type(ty: &Type) -> Option<OwnerQual> {
    match ty.without_mut() {
        Type::Qualified(_, q) => match q {
            OwnerQual::Stack | OwnerQual::Owned | OwnerQual::Shared
            | OwnerQual::Actor | OwnerQual::Guard => Some(q.clone()),
            OwnerQual::Union(_) => None,
            _ => None,
        },
        // Optional<Qualified> — extract the inner qualifier.
        Type::Optional(inner) => qual_of_type(inner.as_ref()),
        _ => None,
    }
}

/// Build the correctly-nested type when applying an inferred qualifier.
/// Handles bare T, T' (tick), T?, and T'? so that the qualifier ends up
/// inside the Optional wrapper rather than outside it.
pub(crate) fn apply_inferred_qual(ty: &Type, qual: OwnerQual) -> Type {
    match ty {
        // T? or T'? — qualifier goes inside the Optional
        Type::Optional(inner) => {
            let inner_base = match inner.as_ref() {
                Type::Qualified(b, _) => b.as_ref().clone(), // strip existing qual (e.g. Owned from T')
                other => other.clone(),
            };
            Type::Optional(Box::new(Type::Qualified(Box::new(inner_base), qual)))
        }
        // T' or T'<group> — replace existing qualifier with the inferred one
        Type::Qualified(inner, _) => Type::Qualified(inner.clone(), qual),
        // bare T
        other => Type::Qualified(Box::new(other.clone()), qual),
    }
}

fn quals_equal(a: &OwnerQual, b: &OwnerQual) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

fn qual_name(q: &OwnerQual) -> &'static str {
    match q {
        OwnerQual::Stack     => "stack",
        OwnerQual::Owned     => "heap",
        OwnerQual::Shared    => "shared",
        OwnerQual::Actor     => "actor",
        OwnerQual::ActorTask => "actor'task",
        OwnerQual::Guard     => "guard",
        OwnerQual::GuardTask => "guard'task",
        OwnerQual::Weak      => "weak",
        OwnerQual::Borrow    => "T&",
        OwnerQual::BorrowMut => "mut T&",
        _                    => "unknown",
    }
}

/// Collect variable names that appear as the object (receiver) of a method call
/// anywhere inside an expression tree, along with the set of method names called
/// on each one (used to detect `task`-method calls for 'actor'task/'guard'task inference).
fn method_receivers(expr: &Expr) -> std::collections::HashMap<String, std::collections::HashSet<String>> {
    let mut out = std::collections::HashMap::new();
    collect_receivers_in_expr(expr, &mut out);
    out
}

fn collect_receivers_in_expr(expr: &Expr, out: &mut std::collections::HashMap<String, std::collections::HashSet<String>>) {
    match &expr.kind {
        ExprKind::MethodCall(obj, method, args) | ExprKind::OptionalMethodCall(obj, method, args) => {
            if let ExprKind::Var(name) = &obj.kind {
                out.entry(name.clone()).or_default().insert(method.clone());
            }
            collect_receivers_in_expr(obj, out);
            for a in args { collect_receivers_in_expr(&a.value, out); }
        }
        ExprKind::Call(callee, args) => {
            collect_receivers_in_expr(callee, out);
            for a in args { collect_receivers_in_expr(&a.value, out); }
        }
        ExprKind::BinOp(_, l, r) => {
            collect_receivers_in_expr(l, out);
            collect_receivers_in_expr(r, out);
        }
        ExprKind::UnaryOp(_, e) | ExprKind::Field(e, _) | ExprKind::OptionalField(e, _) => {
            collect_receivers_in_expr(e, out);
        }
        ExprKind::If(s) => {
            for (cond, body) in &s.branches {
                collect_receivers_in_expr(cond, out);
                for st in body { collect_receivers_in_stmt(st, out); }
            }
            if let Some(eb) = &s.else_body {
                for st in eb { collect_receivers_in_stmt(st, out); }
            }
        }
        ExprKind::Block(stmts) => {
            for st in stmts { collect_receivers_in_stmt(st, out); }
        }
        ExprKind::Array(elems) | ExprKind::Tuple(elems) | ExprKind::Set(elems) => {
            for e in elems { collect_receivers_in_expr(e, out); }
        }
        ExprKind::Assign(target, val) => {
            collect_receivers_in_expr(target, out);
            collect_receivers_in_expr(val, out);
        }
        ExprKind::Else(e, d) | ExprKind::TryElse(e, d) => {
            collect_receivers_in_expr(e, out);
            collect_receivers_in_expr(d, out);
        }
        _ => {}
    }
}

fn collect_receivers_in_stmt(stmt: &Stmt, out: &mut std::collections::HashMap<String, std::collections::HashSet<String>>) {
    match stmt {
        Stmt::Let(s) => { if let Some(v) = &s.value { collect_receivers_in_expr(v, out); } }
        Stmt::Expr(e) | Stmt::Return(crate::ast::ReturnStmt { value: Some(e), .. }) => {
            collect_receivers_in_expr(e, out);
        }
        Stmt::If(s) => {
            for (cond, body) in &s.branches {
                collect_receivers_in_expr(cond, out);
                for st in body { collect_receivers_in_stmt(st, out); }
            }
            if let Some(eb) = &s.else_body {
                for st in eb { collect_receivers_in_stmt(st, out); }
            }
        }
        Stmt::While(s) => {
            collect_receivers_in_expr(&s.condition, out);
            for st in &s.body { collect_receivers_in_stmt(st, out); }
        }
        Stmt::For(s) => {
            collect_receivers_in_expr(&s.iterable, out);
            for st in &s.body { collect_receivers_in_stmt(st, out); }
        }
        Stmt::Match(s) => {
            collect_receivers_in_expr(&s.subject, out);
            for arm in &s.arms {
                match &arm.body {
                    MatchBody::Expr(e) => collect_receivers_in_expr(e, out),
                    MatchBody::Block(stmts) => {
                        for st in stmts { collect_receivers_in_stmt(st, out); }
                    }
                }
            }
        }
        _ => {}
    }
}

/// If `expr` is `self.field_name`, return `field_name`.
fn self_field_name(expr: &Expr) -> Option<&str> {
    if let ExprKind::Field(obj, field) = &expr.kind {
        if let ExprKind::Var(v) = &obj.kind {
            if v == "self" {
                return Some(field.as_str());
            }
        }
    }
    None
}

/// Collect all `self.field` names accessed anywhere in an expression tree.
fn self_field_names_in_expr(expr: &Expr) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    collect_self_fields_in_expr(expr, &mut out);
    out
}

fn collect_self_fields_in_expr(expr: &Expr, out: &mut std::collections::HashSet<String>) {
    if let Some(field) = self_field_name(expr) {
        out.insert(field.to_string());
        return;
    }
    match &expr.kind {
        ExprKind::MethodCall(obj, _, args) | ExprKind::OptionalMethodCall(obj, _, args) => {
            collect_self_fields_in_expr(obj, out);
            for a in args { collect_self_fields_in_expr(&a.value, out); }
        }
        ExprKind::Call(callee, args) => {
            collect_self_fields_in_expr(callee, out);
            for a in args { collect_self_fields_in_expr(&a.value, out); }
        }
        ExprKind::BinOp(_, l, r) => {
            collect_self_fields_in_expr(l, out);
            collect_self_fields_in_expr(r, out);
        }
        ExprKind::UnaryOp(_, e) | ExprKind::Field(e, _) => {
            collect_self_fields_in_expr(e, out);
        }
        ExprKind::If(s) => {
            for (cond, body) in &s.branches {
                collect_self_fields_in_expr(cond, out);
                for st in body { collect_self_fields_in_stmt(st, out); }
            }
            if let Some(eb) = &s.else_body {
                for st in eb { collect_self_fields_in_stmt(st, out); }
            }
        }
        ExprKind::Block(stmts) => {
            for st in stmts { collect_self_fields_in_stmt(st, out); }
        }
        ExprKind::Array(elems) | ExprKind::Tuple(elems) | ExprKind::Set(elems) => {
            for e in elems { collect_self_fields_in_expr(e, out); }
        }
        ExprKind::Assign(target, val) => {
            collect_self_fields_in_expr(target, out);
            collect_self_fields_in_expr(val, out);
        }
        _ => {}
    }
}

fn collect_self_fields_in_stmt(stmt: &Stmt, out: &mut std::collections::HashSet<String>) {
    match stmt {
        Stmt::Let(s) => { if let Some(v) = &s.value { collect_self_fields_in_expr(v, out); } }
        Stmt::Expr(e) | Stmt::Return(crate::ast::ReturnStmt { value: Some(e), .. }) => {
            collect_self_fields_in_expr(e, out);
        }
        Stmt::If(s) => {
            for (cond, body) in &s.branches {
                collect_self_fields_in_expr(cond, out);
                for st in body { collect_self_fields_in_stmt(st, out); }
            }
            if let Some(eb) = &s.else_body {
                for st in eb { collect_self_fields_in_stmt(st, out); }
            }
        }
        Stmt::While(s) => {
            collect_self_fields_in_expr(&s.condition, out);
            for st in &s.body { collect_self_fields_in_stmt(st, out); }
        }
        Stmt::For(s) => {
            collect_self_fields_in_expr(&s.iterable, out);
            for st in &s.body { collect_self_fields_in_stmt(st, out); }
        }
        _ => {}
    }
}
