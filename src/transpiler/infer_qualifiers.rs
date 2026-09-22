use super::Transpiler;
use crate::ast::{BindingKind, Expr, ExprKind, MatchBody, OwnerQual, Stmt, Type};
use super::helpers::collect_var_names;

/// Output of `Transpiler::collect_with_await_signals`: candidate names referenced inside at
/// least one `with` block anywhere in the scanned body (`referenced` — these need lock/
/// interior-mutability semantics, never a plain auto-ref borrow, regardless of awaiting), and
/// the subset of those whose block provably holds across a genuine await (`awaited` — these
/// additionally need the `'task` lock variant). `awaited` is always a subset of `referenced`.
#[derive(Default)]
struct WithSignals {
    referenced: std::collections::HashSet<String>,
    awaited: std::collections::HashSet<String>,
}

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
        self.observed_locals.clear();
        self.task_method_call_vars.clear();

        // Re-seed `'observed`-qualified parameters right after the clear above (not in
        // `seed_param_locals`, which runs *before* this function's first call for the
        // enclosing function body — `emit_body`/`emit_body_optional_last` call
        // `infer_qualifiers` again for nested blocks too, each time clearing
        // `observed_locals` unconditionally, so registering params anywhere upstream of
        // this point would just get wiped out again here). `fn_current_params` is
        // populated once per function (name → declared type, set in `emit_fn` before
        // `emit_body` runs) and stays valid across every nested `infer_qualifiers` call
        // within the same function body, so re-seeding here on every call is redundant
        // but harmless — same idempotent shape as `observed_bare`'s own per-call
        // resolution further down. A bare (no explicit base) `'observed` param defaults
        // to `'actor'observed` via `Transpiler::resolve_bare_observed` — see that
        // function's doc for why params/fields/returns get this simpler default instead
        // of the local-binding usage-based inference the rest of this function
        // implements for bare *local* `'observed` bindings.
        for (name, ty) in self.fn_current_params.clone() {
            if let Type::Qualified(base_qualified, OwnerQual::Observed) = Self::resolve_bare_observed(&ty) {
                if let Type::Qualified(inner, base) = *base_qualified {
                    if let Type::Named(n) = *inner {
                        self.observed_locals.insert(name, (n, base));
                    }
                }
            }
        }

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
                // and type parameters are excluded: the fallback would infer 'inline and
                // emit_param would wrap them incorrectly (Addable'inline → "Addable" instead
                // of impl Addable, Pt'inline bypasses the non-fn alias expansion, etc.).
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
                // T'<group> parameter (includes T'new — Union([Owned, Shared, Actor, Guard]),
                // the candidate-set qualifier that replaced bare tick) — Union members as
                // candidate set. A committed `T'owned` parameter is NOT seeded here — like
                // `'shared`/`'actor`/`'guard`, it's a fixed contract, not inferred (see
                // OwnerQual::is_new's doc comment for why 'owned and 'new are distinguished).
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

        // Bare `T'observed` locals (docs/book.md's "'observed" section, "Qualifier
        // inference for bare `'observed`"): represented at parse time as a single-level
        // `Qualified(Named(n), Observed)` — no base qualifier chosen yet. Judgment call
        // (see the task's own note that this is the one part of the spec most likely to
        // need one): rather than threading a whole new `Observed` member through this
        // file's general candidate-elimination lattice (every `constrain_candidates`/
        // `promote_task_variants`/`resolve_fallback` call site would need to learn about
        // it), a bare-observed local is seeded into the *exact same* `anonymous_vars`/
        // `var_struct_types`/`candidates` machinery as an ordinary bare struct local
        // (using its inner `Named` type) — every existing usage signal (task capture,
        // `def`-call, qualifier demand, `mut` binding, …) narrows it exactly as it would
        // a plain bare `FormModel` local — with exactly one extra restriction applied
        // below: `Shared` is eliminated from the candidate set up front, since
        // `'shared'observed` is never a legal resolution (rejected by the checker for an
        // explicit annotation — see `check_observed_compatibility` — and never a sane
        // *inferred* default either, since a `'shared` value has nothing for `'observed`
        // to notify about). With `Shared` gone, the existing fallback chain
        // (`'owned` > `'shared` > `'actor` > `'atomic` > `'guard`, `docs/book.md` §30)
        // already produces exactly the spec'd defaults for free: a genuine multi-owner
        // usage signal (e.g. passed to something demanding `'actor`) narrows the
        // candidate set down to `{Actor}` (or `{Actor, Guard}`) *before* the fallback
        // even runs, so it's chosen directly as the sole survivor; absent such a signal,
        // the untouched Step-1 size check (`'inline` vs continue) followed by the
        // ordered chain (`'owned` first among what's left) reproduces "otherwise resolve
        // 'inline'observed vs 'owned'observed via the size threshold" exactly. No
        // separate resolution algorithm needed — see the follow-up loop after the main
        // resolution loop below, which just reads back whatever `inferred_qualifiers`
        // this ordinary pipeline already produced and records it in `observed_locals`.
        let mut observed_bare: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        for stmt in stmts {
            collect_bare_observed_lets(stmt, &mut observed_bare);
        }
        for (name, struct_name) in &observed_bare {
            anonymous_vars.insert(name.clone());
            var_struct_types.insert(name.clone(), struct_name.clone());
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

        // Bare `'observed` locals: eliminate `Shared` from the candidate set up front —
        // see the long comment above `observed_bare`'s collection for why.
        for name in observed_bare.keys() {
            constrain_candidates(
                &mut candidates, name,
                &[OwnerQual::Inline, OwnerQual::Owned, OwnerQual::Actor, OwnerQual::ActorTask,
                  OwnerQual::Guard, OwnerQual::GuardTask, OwnerQual::Atomic],
                &alias_of,
            );
        }

        // Bare `'observed` locals: multi-owner usage signal ("passed to something that
        // demands `'actor`/`'guard`") — the spec's example for defaulting toward
        // `'actor'observed`. Real usage of an `'observed` local's base value always goes
        // through `.value` (`spawn_actor(c.value)`, never `spawn_actor(c)` — the bare
        // name refers to the *wrapper*, not the base value) — a shape the general
        // signal-collection walk below (`walk_expr_for_qualifiers`) does not recognize,
        // since every one of its call-site-demand checks matches only a bare
        // `ExprKind::Var` argument (see e.g. its own `Call` arm just below). Rather than
        // teach that whole general walk (and every other bare-`Var`-keyed signal check
        // alongside it — task captures, `with` blocks, …) about this one extra shape,
        // this is a narrow, dedicated scan for exactly the shape the spec's own example
        // needs: `<var>.value` passed as a direct call argument, at a statement's top
        // level, to a function whose declared parameter at that position demands
        // `'actor`/`'guard`. A deeper expression nesting (inside a binary op, a nested
        // call, a closure body, …) is not covered — a real, intentional scope
        // limitation of the bare-`'observed` inference judgment call (see this
        // session's report).
        for stmt in stmts {
            scan_observed_bare_multi_owner_signal(self, stmt, &observed_bare, &mut candidates, &alias_of);
        }

        // `mut` binding → mutation signal at declaration site: eliminates Shared.
        for var_name in &mut_bindings {
            constrain_candidates(
                &mut candidates, var_name,
                &[OwnerQual::Inline, OwnerQual::Owned, OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask],
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

        // Local live-range analysis (extends the task-method-call heuristic above): a `with`
        // block conceptually holds its named value's lock for its whole span, so a genuine
        // await anywhere in that span — even one utterly unrelated to the guarded value, and
        // even when nothing called on the guarded value inside the block is itself declared
        // `task` — is a positive signal that the 'task lock variant is required. Scans the
        // whole body (including inside nested task/closure literals, which is exactly where a
        // captured actor/guard value is normally used) for such blocks; every named candidate
        // whose block provably holds across an await is fed into the same disambiguation signal
        // (`task_method_call_vars`) `constrain_task_captures` already populates above, whether or
        // not that var was also captured by a task/closure. See docs/qualifiers.md's "Inferring
        // 'actor'task/'guard'task from task-method calls" for the full design and worked example.
        let mut with_signals = WithSignals::default();
        self.collect_with_await_signals(stmts, &anonymous_vars, &var_struct_types, &mut with_signals);
        for name in &with_signals.referenced {
            // A `with` block always requires lock/interior-mutability semantics — never a
            // plain auto-ref borrow — regardless of whether this particular usage also
            // proves an await (mirrors the task-capture "storage signal" handling above).
            constrain_candidates(
                &mut candidates, name,
                &[OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask],
                &alias_of,
            );
            if auto_ref_param_vars.contains(name.as_str()) {
                has_qualifier_constraint.insert(name.clone());
            }
        }
        for name in &with_signals.awaited {
            promote_task_variants(&mut candidates, name, &alias_of);
            self.task_method_call_vars.insert(name.clone());
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
                    self.push_error(line, 0, format!(
                        "`{}` has no valid qualifier — usage constraints are incompatible\n  \
                         fix: annotate `{}` explicitly",
                        var_name, var_name
                    ));
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
                        &remaining, false, type_size, self.config.inline_auto_bytes,
                        self.config.mode == crate::transpiler::TranspileMode::Strict,
                    ) {
                        self.inferred_qualifiers.insert(var_name.clone(), q);
                    }
                }
            }
        }

        // Bare `'observed` locals: read back whatever the ordinary bare-struct
        // resolution above just produced (`Inline`/`Owned`/`Actor`/`ActorTask`/`Guard`/
        // `GuardTask` — `Shared` was excluded up front, so it never appears here) and
        // record it in `observed_locals` for `emit_let`/`emit_methods` to consume.
        // `ActorTask`/`GuardTask` are treated the same as `Actor`/`Guard` by the
        // consumers of this map (tokio-lock variants of the same representation).
        for (name, struct_name) in &observed_bare {
            if let Some(q) = self.inferred_qualifiers.get(name).cloned() {
                self.observed_locals.insert(name.clone(), (struct_name.clone(), q));
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
                // For 'shared/'actor/'guard demands, 'inline and 'owned are also compatible
                // because a plain T or Box<T> can be wrapped at the call site.
                if let ExprKind::Var(fn_name) = &callee.kind {
                    let param_types = self.fn_sigs.get(fn_name.as_str()).cloned();
                    if let Some(param_types) = param_types {
                        for (i, arg) in args.iter().enumerate() {
                            let Some(param_ty) = param_types.get(i) else { continue };
                            let Some(demanded) = qual_of_type(param_ty) else { continue };
                            let ExprKind::Var(var_name) = &arg.value.kind else { continue };
                            if anonymous_vars.contains(var_name.as_str()) {
                                // Cross-function propagation (docs/qualifiers.md's "Inferring
                                // 'actor'task/'guard'task"): the callee's own signature already
                                // demands the 'task lock variant (either because it was inferred
                                // from the callee's body — see `collect_with_await_signals` below
                                // — or explicitly annotated). `coercible_from(ActorTask/GuardTask)`
                                // only accepts the exact 'task variant, so without first widening
                                // this argument's own candidate set the same way a task/closure
                                // capture would (`promote_task_variants`), a caller whose value is
                                // still only a plain `Actor`/`Guard` candidate would have its
                                // candidate set intersected to empty here — a spurious "no valid
                                // qualifier" conflict instead of the caller silently adopting the
                                // 'task variant too (the two lock types are different concrete Rust
                                // types; the caller MUST agree with the callee, it cannot just
                                // decline). This is the only place a plain `Actor`/`Guard` candidate
                                // is ever widened outside an actual task/closure capture — it never
                                // fires unless some callee has already proven (or been told) it
                                // needs the async lock, so it can never fire from mere absence of
                                // proof (see the module doc comment's "direction 2" design note).
                                if matches!(demanded, OwnerQual::ActorTask | OwnerQual::GuardTask) {
                                    promote_task_variants(candidates, var_name, alias_of);
                                }
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
                            // `method_is_req_or_task` also consults `trait_default_mutating`
                            // (via `struct_protocols`) on top of `struct_req_methods` — needed
                            // for a header-only-declared trait's methods (a user `trait ... :
                            // req/def`, or the built-in `Introspect`, see its doc comment in
                            // `mod.rs`), which never populate `struct_req_methods` themselves.
                            // Using the narrower `struct_req_methods`-only check here (as this
                            // used to) reported a false "not declared mut" diagnostic for such
                            // methods even though `emit_top.rs`'s own req/def gate — the one
                            // that actually decides whether transpilation fails — already knew
                            // better; this call keeps both checks in sync.
                            self.method_is_req_or_task(struct_name, method)
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
                                self.push_error(line, 0, format!(
                                    "parameter `{}` is immutable but `{}` is a `def` method \
                                     — declare `mut {} n`",
                                    var_name, method, var_name
                                ));
                            }
                            constrain_candidates(
                                candidates, var_name,
                                &[OwnerQual::Inline, OwnerQual::Owned, OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask],
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
                                &[OwnerQual::Inline, OwnerQual::Owned, OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask],
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

    /// Local live-range analysis for the `'actor'task`/`'guard'task` inference (see
    /// `docs/qualifiers.md`'s "Inferring `'actor'task`/`'guard'task` from task-method calls").
    ///
    /// A `with name[, name...]:` block conceptually holds each named value's lock for its
    /// entire span (see `docs/scoped-access-blocks.md`) — so a genuine await *anywhere* in
    /// that span, not just a call to a `task`-declared method on the guarded value itself, is
    /// proof the async lock variant is required. This scans `stmts` for every `with` block
    /// (recursing into `if`/`while`/`for`/`match`/`try`/`guard`/nested `with`/loop bodies —
    /// the same boundary `ast::with_block_mutates` already uses for its own read/write scan —
    /// and also into task/closure literal bodies, since that's exactly where a captured
    /// actor/guard value is normally used) and, for each one found, checks whether its own
    /// body (bounded the same way, but *not* crossing into a further-nested task/closure
    /// literal — that spawns its own, separately scheduled async context and does not block
    /// the current span) contains an unambiguous await point via `stmts_have_await`.
    ///
    /// Deliberately narrow, matching this module's "conservative toward sync" design: only
    /// three syntactic forms count as an await point — an explicit `wait(...)`, a `join […]`,
    /// or a direct call into an already-known `task`-declared function/method (`task_fns` /
    /// `struct_task_methods`, both populated by pre-scan before any body is walked). A
    /// blocking `.value`/`.wait` on a spawned task handle is a real await too, but recognizing
    /// it needs type information (`task_vars` and friends) this early pre-pass doesn't have —
    /// missing it just means staying on the plain sync variant, the safe direction, not a
    /// silent wrong answer; see docs/qualifiers.md's residual-gap paragraph.
    ///
    /// Before/after:
    /// ```boring
    /// struct Counter:
    ///     var int value = 0
    ///     def inc():             # plain `def`, not `task` — today's own heuristic alone
    ///         value += 1          # would leave `c` on the plain sync variant here
    ///
    /// task def void worker(Counter c):
    ///     with c:
    ///         c.inc()             # not a `task` method call — the OLD heuristic sees nothing
    ///         wait(Duration.fromMillis(100))   # …but this NEW scan finds the await in the
    ///                                          # same held span, so `c` still infers
    ///                                          # 'actor'task, matching `docs/qualifiers.md`'s
    ///                                          # example that used to need an explicit annotation.
    /// ```
    fn collect_with_await_signals(
        &self,
        stmts: &[Stmt],
        anonymous_vars: &std::collections::HashSet<String>,
        var_struct_types: &std::collections::HashMap<String, String>,
        out: &mut WithSignals,
    ) {
        for stmt in stmts {
            if let Stmt::With(w) = stmt {
                let has_await = self.stmts_have_await(&w.body, var_struct_types);
                for name in &w.names {
                    if !anonymous_vars.contains(name.as_str()) { continue; }
                    // Any `with`-block usage at all requires lock/interior-mutability
                    // semantics — incompatible with a plain auto-ref borrow — regardless
                    // of whether this particular block also proves an await.
                    out.referenced.insert(name.clone());
                    if has_await {
                        out.awaited.insert(name.clone());
                    }
                }
            }
            self.collect_with_await_signals_stmt(stmt, anonymous_vars, var_struct_types, out);
        }
    }

    fn collect_with_await_signals_stmt(
        &self,
        stmt: &Stmt,
        anonymous_vars: &std::collections::HashSet<String>,
        var_struct_types: &std::collections::HashMap<String, String>,
        out: &mut WithSignals,
    ) {
        let b = |body: &[Stmt], out: &mut WithSignals| {
            self.collect_with_await_signals(body, anonymous_vars, var_struct_types, out)
        };
        match stmt {
            Stmt::With(w) => b(&w.body, out),
            Stmt::If(s) => {
                for (_, body) in &s.branches { b(body, out); }
                if let Some(eb) = &s.else_body { b(eb, out); }
            }
            Stmt::IfLet(s) => {
                b(&s.then_body, out);
                for br in &s.elif_branches { b(&br.body, out); }
                if let Some(eb) = &s.else_body { b(eb, out); }
            }
            Stmt::While(s) => b(&s.body, out),
            Stmt::WhileLet(s) => b(&s.body, out),
            Stmt::DoWhile(s) => b(&s.body, out),
            Stmt::Loop(s) => b(&s.body, out),
            Stmt::For(s) => b(&s.body, out),
            Stmt::Match(s) => {
                for arm in &s.arms {
                    if let MatchBody::Block(body) = &arm.body { b(body, out); }
                }
            }
            Stmt::Try(s) => {
                b(&s.body, out);
                for c in &s.catch_clauses { b(&c.body, out); }
            }
            Stmt::Guard(s) => b(&s.else_body, out),
            Stmt::Defer(body) => b(body, out),
            Stmt::KernelBlock(s) => b(&s.body, out),
            // `let`/`return`/expr statements: a `with` block can't appear inline in an
            // expression position (it's parsed only as a statement), but it CAN be nested
            // inside a task/closure literal or an if/match *expression* carried by one of
            // these — recurse into the expression to find those.
            Stmt::Let(s) => { if let Some(v) = &s.value { self.collect_with_await_signals_expr(v, anonymous_vars, var_struct_types, out); } }
            Stmt::Expr(e) => self.collect_with_await_signals_expr(e, anonymous_vars, var_struct_types, out),
            Stmt::Return(r) => { if let Some(v) = &r.value { self.collect_with_await_signals_expr(v, anonymous_vars, var_struct_types, out); } }
            _ => {}
        }
    }

    fn collect_with_await_signals_expr(
        &self,
        expr: &Expr,
        anonymous_vars: &std::collections::HashSet<String>,
        var_struct_types: &std::collections::HashMap<String, String>,
        out: &mut WithSignals,
    ) {
        let b = |body: &[Stmt], out: &mut WithSignals| {
            self.collect_with_await_signals(body, anonymous_vars, var_struct_types, out)
        };
        let e = |ex: &Expr, out: &mut WithSignals| {
            self.collect_with_await_signals_expr(ex, anonymous_vars, var_struct_types, out)
        };
        match &expr.kind {
            ExprKind::Task(inner) => e(inner, out),
            ExprKind::TaskWithTimeout(dur, inner) => { e(dur, out); e(inner, out); }
            ExprKind::Closure(_, _, body, _, _) => match body {
                crate::ast::ClosureBody::Expr(ex) => e(ex, out),
                crate::ast::ClosureBody::Block(stmts) => b(stmts, out),
            },
            ExprKind::Block(stmts) | ExprKind::Do(stmts) => b(stmts, out),
            ExprKind::If(s) => {
                for (c, body) in &s.branches { e(c, out); b(body, out); }
                if let Some(eb) = &s.else_body { b(eb, out); }
            }
            ExprKind::Match(s) => {
                e(&s.subject, out);
                for arm in &s.arms {
                    match &arm.body {
                        MatchBody::Expr(ex) => e(ex, out),
                        MatchBody::Block(body) => b(body, out),
                    }
                }
            }
            ExprKind::TryElseBlock(body, els) => { b(body, out); b(els, out); }
            ExprKind::Call(callee, args) | ExprKind::MethodCall(callee, _, args)
            | ExprKind::OptionalMethodCall(callee, _, args) | ExprKind::GenericCall(callee, _, args)
            | ExprKind::Pipe(callee, _, args) => {
                e(callee, out);
                for a in args { e(&a.value, out); }
            }
            ExprKind::BinOp(_, l, r) | ExprKind::Assign(l, r) | ExprKind::QuestionAssign(l, r)
            | ExprKind::Else(l, r) | ExprKind::TryElse(l, r) => { e(l, out); e(r, out); }
            ExprKind::UnaryOp(_, inner) | ExprKind::Field(inner, _) | ExprKind::OptionalField(inner, _)
            | ExprKind::Cast(inner, _) => e(inner, out),
            ExprKind::Index(o, i) => { e(o, out); e(i, out); }
            ExprKind::Array(es) | ExprKind::Tuple(es) | ExprKind::Set(es) => {
                for ex in es { e(ex, out); }
            }
            _ => {}
        }
    }

    /// Bounded await-point scan for a `with` block's own body (the "held span"). Same
    /// recursive boundary as `collect_with_await_signals_stmt` — control flow within the
    /// SAME synchronous span — except it must NOT cross into a nested `task`/`TaskWithTimeout`/
    /// `Closure` literal: that spawns a new, separately scheduled async context, so an await
    /// inside it does not block whatever is holding the `with` block's lock.
    fn stmts_have_await(&self, stmts: &[Stmt], var_struct_types: &std::collections::HashMap<String, String>) -> bool {
        stmts.iter().any(|s| self.stmt_has_await(s, var_struct_types))
    }

    fn stmt_has_await(&self, stmt: &Stmt, vst: &std::collections::HashMap<String, String>) -> bool {
        let e = |ex: &Expr| self.expr_has_await(ex, vst);
        let b = |body: &[Stmt]| self.stmts_have_await(body, vst);
        match stmt {
            Stmt::Wait(_, _) => true,
            Stmt::Let(s) => s.value.as_ref().is_some_and(e),
            Stmt::LetDestructure(s) => e(&s.value),
            Stmt::Return(r) => r.value.as_ref().is_some_and(e),
            Stmt::Throw(t) => t.value.as_ref().is_some_and(e),
            Stmt::Expr(ex) => e(ex),
            Stmt::If(s) => s.branches.iter().any(|(c, body)| e(c) || b(body))
                || s.else_body.as_ref().is_some_and(|body| b(body)),
            Stmt::IfLet(s) => {
                s.clauses.iter().any(|c| self.cond_clause_has_await(c, vst))
                    || b(&s.then_body)
                    || s.elif_branches.iter().any(|br| {
                        br.clauses.iter().any(|c| self.cond_clause_has_await(c, vst)) || b(&br.body)
                    })
                    || s.else_body.as_ref().is_some_and(|body| b(body))
            }
            Stmt::Match(s) => e(&s.subject) || s.arms.iter().any(|arm| {
                arm.guard.as_ref().is_some_and(e) || match &arm.body {
                    MatchBody::Expr(ex) => e(ex),
                    MatchBody::Block(body) => b(body),
                }
            }),
            Stmt::While(s) => e(&s.condition) || b(&s.body),
            Stmt::WhileLet(s) => e(&s.value) || b(&s.body),
            Stmt::DoWhile(s) => b(&s.body) || e(&s.condition),
            Stmt::Loop(s) => b(&s.body),
            Stmt::For(s) => e(&s.iterable) || b(&s.body),
            Stmt::Guard(s) => {
                let cond_hit = match &s.cond {
                    crate::ast::GuardCond::Expr(ex) => e(ex),
                    crate::ast::GuardCond::Clauses(cs) => cs.iter().any(|c| self.cond_clause_has_await(c, vst)),
                };
                cond_hit || b(&s.else_body)
            }
            Stmt::Try(s) => b(&s.body) || s.catch_clauses.iter().any(|c| b(&c.body)),
            Stmt::Defer(body) => b(body),
            // Nested `with` — still the same held span (see `ast::with_block_mutates`'s
            // identical treatment for its own read/write scan).
            Stmt::With(s) => b(&s.body),
            Stmt::Yield(ex, _) => e(ex),
            Stmt::Break(_, v) => v.as_ref().is_some_and(e),
            Stmt::KernelBlock(s) => b(&s.body),
            _ => false,
        }
    }

    fn cond_clause_has_await(&self, c: &crate::ast::CondClause, vst: &std::collections::HashMap<String, String>) -> bool {
        match c {
            crate::ast::CondClause::Expr(ex) => self.expr_has_await(ex, vst),
            crate::ast::CondClause::Let(_, v) | crate::ast::CondClause::LetPat(_, v) => self.expr_has_await(v, vst),
        }
    }

    fn expr_has_await(&self, expr: &Expr, vst: &std::collections::HashMap<String, String>) -> bool {
        let e = |ex: &Expr| self.expr_has_await(ex, vst);
        let b = |body: &[Stmt]| self.stmts_have_await(body, vst);
        match &expr.kind {
            // `join […]` always awaits every listed future.
            ExprKind::JoinAll(_) => true,
            // A direct call to an already-known `task`-declared free function, not spawned via
            // the `task` keyword (that's `ExprKind::Task`, handled below), is an implicit await
            // in the emitted Rust — see `emit_expr.rs`'s `is_task`/`self.task_fns` check.
            ExprKind::Call(callee, args) => {
                let direct = matches!(&callee.kind, ExprKind::Var(n) if self.task_fns.contains(n.as_str()));
                direct || e(callee) || args.iter().any(|a| e(&a.value))
            }
            // Same, for a directly-called (not spawned) `task`-declared method — this is the
            // existing "called-method-is-itself-task" signal, just reachable here too so an
            // *unrelated* receiver's task-method call still counts as an await in this span.
            // Only resolved for a bare-variable receiver whose struct type this pre-pass already
            // tracks (`var_struct_types`) — a `self.field` receiver or an explicitly-qualified
            // variable isn't resolvable this early; missing it just stays conservative (no
            // upgrade), never a false positive.
            ExprKind::MethodCall(recv, method, args) | ExprKind::OptionalMethodCall(recv, method, args) => {
                let direct = matches!(&recv.kind, ExprKind::Var(n) if vst.get(n.as_str())
                    .is_some_and(|struct_name| self.struct_task_methods.contains(&format!("{}::{}", struct_name, method))));
                direct || e(recv) || args.iter().any(|a| e(&a.value))
            }
            ExprKind::BinOp(_, l, r) | ExprKind::Assign(l, r) | ExprKind::QuestionAssign(l, r)
            | ExprKind::Else(l, r) | ExprKind::TryElse(l, r) => e(l) || e(r),
            ExprKind::UnaryOp(_, inner) | ExprKind::Field(inner, _) | ExprKind::OptionalField(inner, _)
            | ExprKind::Cast(inner, _) => e(inner),
            ExprKind::Index(o, i) => e(o) || e(i),
            ExprKind::Array(es) | ExprKind::Tuple(es) | ExprKind::Set(es) => es.iter().any(e),
            ExprKind::If(s) => s.branches.iter().any(|(c, body)| e(c) || b(body))
                || s.else_body.as_ref().is_some_and(|body| b(body)),
            ExprKind::Match(s) => e(&s.subject) || s.arms.iter().any(|arm| match &arm.body {
                MatchBody::Expr(ex) => e(ex),
                MatchBody::Block(body) => b(body),
            }),
            ExprKind::Block(stmts) | ExprKind::Do(stmts) => b(stmts),
            ExprKind::Loop(s) => b(&s.body),
            ExprKind::GenericCall(callee, _, args) | ExprKind::Pipe(callee, _, args) => {
                e(callee) || args.iter().any(|a| e(&a.value))
            }
            // New async contexts — spawned separately, do not block the current span.
            ExprKind::Task(_) | ExprKind::TaskWithTimeout(_, _) | ExprKind::Closure(_, _, _, _, _) => false,
            _ => false,
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
            // `mut Point p` wraps `f.ty` in `Type::Mut` (docs/book.md
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
                    // so 'inline is always preferred when available.
                    let type_size = target_fields.get(field_name.as_str())
                        .and_then(|tn| self.type_sizes.get(tn.as_str()))
                        .copied();
                    match resolve_fallback(
                        &remaining, true, type_size, self.config.inline_auto_bytes,
                        self.config.mode == crate::transpiler::TranspileMode::Strict,
                    ) {
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
                                // Same cross-function widening as the local-variable Call case
                                // in `walk_expr_for_qualifiers` — see its comment for why this
                                // is needed before intersecting, not just for symmetry.
                                if matches!(demanded, OwnerQual::ActorTask | OwnerQual::GuardTask) {
                                    promote_task_variants(candidates, field_name, alias_of);
                                }
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
                                &[OwnerQual::Inline, OwnerQual::Owned, OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask],
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
                                    self.push_error(expr.line, expr.col, format!(
                                        "qualifier '{}' for `{}` is not allowed here\n  \
                                         → parameter {} of `{}` accepts only: {}\n  \
                                         fix: annotate `{}` with one of the listed qualifiers",
                                        qual_name(&arg_qual), var_name,
                                        i + 1, fn_name, allowed.join("|"),
                                        var_name
                                    ));
                                }
                            }

                            // Body-compatibility check: Union-typed argument passed where a concrete
                            // qualifier is demanded — verify the demanded qualifier is in the union.
                            if let Some(demanded) = qual_of_type(param_ty) {
                                if let Some(arg_union) = self.var_union(var_name) {
                                    if !arg_union.iter().any(|m| quals_equal(m, &demanded)) {
                                        let union_s: Vec<&str> = arg_union.iter().map(|q| qual_name(q)).collect();
                                        self.push_error(expr.line, expr.col, format!(
                                            "`{}` has qualifier constraint '{}'\n  \
                                             → this call demands '{}' which is outside the constraint\n  \
                                             fix: change the qualifier constraint on `{}` to include '{}', \
                                             or pick a concrete qualifier",
                                            var_name, union_s.join("|"),
                                            qual_name(&demanded), var_name, qual_name(&demanded)
                                        ));
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
                // Advisory only — a successfully-inferred qualifier with a suggestion to
                // annotate it explicitly, not a conflict. Unlike the other 4 `eprintln!`
                // sites this module used to have (see the audit finding this fixes), this
                // one is legitimately non-fatal: `push_warning` (not `push_error`) is the
                // right severity — promoting it to a hard error would fail every build with
                // an unannotated-but-successfully-inferred parameter, which is the common
                // case, not a bug.
                self.push_warning(line, 0, format!(
                    "parameter `{}` is always used as '{}' in this body — \
                     consider annotating it explicitly to make the contract clear at call sites",
                    param_name, qual_name(inferred)
                ));
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

/// For 'shared/'actor/'guard demands, 'inline and 'owned are also acceptable
/// because a plain T or Box<T> can be wrapped at the call site.
/// For 'inline/'owned demands, only the exact qualifier is accepted.
fn coercible_from(demanded: OwnerQual) -> Vec<OwnerQual> {
    match demanded {
        OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard | OwnerQual::Atomic =>
            vec![OwnerQual::Inline, OwnerQual::Owned, demanded],
        // Universal immutable borrow: any qualifier is accepted — no constraint on caller.
        OwnerQual::Borrow => all_qualifiers(),
        // Universal mutable borrow: any mutable qualifier ('shared excluded).
        OwnerQual::BorrowMut =>
            vec![OwnerQual::Inline, OwnerQual::Owned, OwnerQual::Actor, OwnerQual::ActorTask, OwnerQual::Guard, OwnerQual::GuardTask],
        _ => vec![demanded],
    }
}

fn all_qualifiers() -> Vec<OwnerQual> {
    vec![
        OwnerQual::Inline,
        OwnerQual::Owned,
        OwnerQual::Shared,
        OwnerQual::Actor,
        OwnerQual::Guard,
    ]
}

/// Priority-ordered fallback when multiple qualifier candidates remain after constraint
/// elimination.
///
/// 1. If `Inline` ∈ candidates:
///    - struct field (any binding) → `'inline` (bytes are part of parent allocation)
///    - local variable, sizeof(T) ≤ inline_auto_bytes → `'inline`
///    - type too large, and size-based auto-boxing applies (`size_boxing_applies`,
///      strict mode only — see `docs/transpilation-modes.md` "Size-based auto-boxing
///      (strict mode only)") → skip `'inline`, go to ordered chain
///    - type too large, but size-based auto-boxing does NOT apply (managed mode) →
///      stay `'inline` regardless of size
///
/// 2. Ordered chain: `'owned` > `'shared` > `'actor`(/`'actor'task`) > `'guard`(/`'guard'task`)
///
/// `size_boxing_applies` is `false` in `--mode managed`: that mode's own `'owned` →
/// `Arc<Mutex<T>>`/`RefCell<T>` promotion is keyed on an explicit qualifier, never a bare
/// name (see `promote_bare_return_ty`, `src/transpiler/emit_top.rs`), so a bare oversized
/// local must stay `'inline` too — otherwise the local's own managed-wrapper type could
/// disagree with a bare, unwrapped enclosing function return type (or an unwrapped bare
/// constructor call initializing it), a real `cargo build` mismatch confirmed and documented
/// in `docs/qualifiers.md` "Managed mode and a bare oversized local variable" before this fix.
fn resolve_fallback(
    candidates: &[OwnerQual],
    is_struct_field: bool,
    type_size: Option<usize>,
    inline_auto_bytes: usize,
    size_boxing_applies: bool,
) -> Option<OwnerQual> {
    let has = |q: &OwnerQual| candidates.iter().any(|c| quals_equal(c, q));
    let fits = !size_boxing_applies || type_size.is_none_or(|s| s <= inline_auto_bytes);

    // Ordered chain: 'owned > 'shared > 'actor(/'actor'task) > 'atomic > 'guard(/'guard'task).
    // The 'task variant is checked first at each slot so that it wins when it's the one
    // that survived constraint elimination (e.g. after `disambiguate_task_variant`) —
    // by that point at most one of {Actor, ActorTask} and one of {Guard, GuardTask} remain.
    //
    // 'atomic sits after 'actor(/'actor'task) and before 'guard(/'guard'task) — but this
    // position is inert for default-selection purposes: 'actor(/'actor'task) is checked
    // first and always wins the tie-break whenever {Actor, Guard, Atomic} (or any subset
    // containing Actor) remain candidates simultaneously, exactly like 'guard already never
    // wins today. 'atomic is reachable only via an explicit `x'atomic` annotation or an
    // explicit call-site demand (a parameter typed `T'atomic`) — see
    // docs/qualifiers.md's `'atomic` section and this file's `mod tests` below, which
    // pins this inertness down as a regression test.
    fn tail_pick(has: &dyn Fn(&OwnerQual) -> bool) -> Option<OwnerQual> {
        if has(&OwnerQual::Owned) { return Some(OwnerQual::Owned); }
        if has(&OwnerQual::Shared) { return Some(OwnerQual::Shared); }
        if has(&OwnerQual::ActorTask) { return Some(OwnerQual::ActorTask); }
        if has(&OwnerQual::Actor) { return Some(OwnerQual::Actor); }
        if has(&OwnerQual::Atomic) { return Some(OwnerQual::Atomic); }
        if has(&OwnerQual::GuardTask) { return Some(OwnerQual::GuardTask); }
        if has(&OwnerQual::Guard) { return Some(OwnerQual::Guard); }
        None
    }

    // Step 1: 'inline — struct field of any binding, or small local variable.
    if has(&OwnerQual::Inline) {
        if is_struct_field {
            return Some(OwnerQual::Inline);
        }
        if fits {
            return Some(OwnerQual::Inline);
        }
        return tail_pick(&has);
    }

    // Step 2: 'inline not in candidates — first from the ordered chain.
    tail_pick(&has)
}

/// Candidate set for T'new variables: indirection is certain, kind is inferred.
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
            // NOTE: only `'new` (Union([Owned, Shared, Actor, Guard])) is tick-like here —
            // a committed `'owned` is a fixed contract, like `'shared`/`'actor`/`'guard`,
            // and must NOT be seeded for inference (see OwnerQual::is_new's doc comment).
            let is_tick = match &s.ty {
                Some(Type::Qualified(_, q)) if q.is_new() => true,
                Some(Type::Optional(inner)) => matches!(inner.as_ref(), Type::Qualified(_, q) if q.is_new()),
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
                // T'new or T'new? binding: indirection hint, restricts to {Owned, Shared, Actor, Guard}.
                if is_tick {
                    tick_bindings.insert(s.name.clone());
                }
                // `mut` binding: mutation signal at declaration site.
                if s.binding == BindingKind::Mut {
                    mut_bindings.insert(s.name.clone());
                }
                if let Some(val) = &s.value {
                    // `new Constructor()` on RHS without arena: treat as tick binding
                    // (infer excluding 'inline), same as T'new.
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

/// Narrow, dedicated scan for the bare-`'observed` "multi-owner usage" signal — see
/// the long comment at its call site in `infer_qualifiers` for why this exists as its
/// own small function instead of extending the general `walk_expr_for_qualifiers`.
/// Only recognizes a top-level call-statement shape: `fn_name(..., var.value, ...)`
/// where `fn_name`'s already-known signature (`self.fn_sigs`) demands `'actor`/
/// `'guard`(`'task`) at that argument position. Recurses into the same nested-block
/// `Stmt` shapes `collect_bare_observed_lets`/`collect_anonymous_vars` already do.
fn scan_observed_bare_multi_owner_signal(
    transpiler: &Transpiler,
    stmt: &Stmt,
    observed_bare: &std::collections::HashMap<String, String>,
    candidates: &mut std::collections::HashMap<String, Vec<OwnerQual>>,
    alias_of: &std::collections::HashMap<String, String>,
) {
    let mut scan_call = |callee: &Expr, args: &[crate::ast::Arg]| {
        let ExprKind::Var(fn_name) = &callee.kind else { return };
        let Some(param_types) = transpiler.fn_sigs.get(fn_name.as_str()) else { return };
        for (i, arg) in args.iter().enumerate() {
            let Some(param_ty) = param_types.get(i) else { continue };
            let Some(demanded) = qual_of_type(param_ty) else { continue };
            if !matches!(demanded, OwnerQual::Actor | OwnerQual::ActorTask | OwnerQual::Guard | OwnerQual::GuardTask) {
                continue;
            }
            if let ExprKind::Field(inner, field) = &arg.value.kind {
                if field == "value" {
                    if let ExprKind::Var(name) = &inner.kind {
                        if observed_bare.contains_key(name.as_str()) {
                            constrain_candidates(candidates, name, &[OwnerQual::Actor, OwnerQual::Guard], alias_of);
                        }
                    }
                }
            }
        }
    };
    match stmt {
        Stmt::Expr(e) => {
            match &e.kind {
                ExprKind::Call(callee, args) => scan_call(callee, args),
                ExprKind::MethodCall(_, _, _) => {} // receiver-based, not a plain fn-name call
                _ => {}
            }
        }
        Stmt::Let(s) => {
            if let Some(ExprKind::Call(callee, args)) = s.value.as_ref().map(|v| &v.kind) {
                scan_call(callee, args);
            }
        }
        Stmt::If(s) => {
            for (_, body) in &s.branches {
                for st in body { scan_observed_bare_multi_owner_signal(transpiler, st, observed_bare, candidates, alias_of); }
            }
            if let Some(else_body) = &s.else_body {
                for st in else_body { scan_observed_bare_multi_owner_signal(transpiler, st, observed_bare, candidates, alias_of); }
            }
        }
        Stmt::While(s) => {
            for st in &s.body { scan_observed_bare_multi_owner_signal(transpiler, st, observed_bare, candidates, alias_of); }
        }
        Stmt::For(s) => {
            for st in &s.body { scan_observed_bare_multi_owner_signal(transpiler, st, observed_bare, candidates, alias_of); }
        }
        Stmt::Match(s) => {
            for arm in &s.arms {
                if let MatchBody::Block(stmts) = &arm.body {
                    for st in stmts { scan_observed_bare_multi_owner_signal(transpiler, st, observed_bare, candidates, alias_of); }
                }
            }
        }
        _ => {}
    }
}

/// Collects `let`/`mut`/`var` locals whose declared type is a *bare* `'observed`
/// annotation (`Type::Qualified(Type::Named(n), OwnerQual::Observed)`, single-level —
/// no base qualifier chosen yet, e.g. `FormModel'observed model = FormModel()`) into
/// `out` (var name → base struct type name). Mirrors `collect_anonymous_vars`'s
/// recursive shape (same `Stmt` cases) rather than reusing it directly, since it needs
/// a different, narrower match on `s.ty` and populates a different output shape.
fn collect_bare_observed_lets(stmt: &Stmt, out: &mut std::collections::HashMap<String, String>) {
    match stmt {
        Stmt::Let(s) => {
            if let Some(Type::Qualified(inner, OwnerQual::Observed)) = s.ty.as_ref().map(Type::without_mut) {
                if let Type::Named(n) = inner.as_ref() {
                    out.insert(s.name.clone(), n.clone());
                }
            }
        }
        Stmt::If(s) => {
            for (_, body) in &s.branches {
                for st in body { collect_bare_observed_lets(st, out); }
            }
            if let Some(else_body) = &s.else_body {
                for st in else_body { collect_bare_observed_lets(st, out); }
            }
        }
        Stmt::While(s) => {
            for st in &s.body { collect_bare_observed_lets(st, out); }
        }
        Stmt::For(s) => {
            for st in &s.body { collect_bare_observed_lets(st, out); }
        }
        Stmt::Match(s) => {
            for arm in &s.arms {
                if let MatchBody::Block(stmts) = &arm.body {
                    for st in stmts { collect_bare_observed_lets(st, out); }
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
            OwnerQual::Inline | OwnerQual::Owned | OwnerQual::Shared
            | OwnerQual::Actor | OwnerQual::ActorTask
            | OwnerQual::Guard | OwnerQual::GuardTask | OwnerQual::Atomic => Some(q.clone()),
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
        OwnerQual::Inline    => "inline",
        OwnerQual::Owned     => "owned",
        OwnerQual::Shared    => "shared",
        OwnerQual::Actor     => "actor",
        OwnerQual::ActorTask => "actor'task",
        OwnerQual::Guard     => "guard",
        OwnerQual::GuardTask => "guard'task",
        OwnerQual::Atomic    => "atomic",
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
        // Same rationale as `collect_vars_in_stmt`'s `Stmt::With` arm (helpers.rs) —
        // a `with c:` block's body is ordinary nested code still calling methods on `c`.
        Stmt::With(w) => {
            for st in &w.body { collect_receivers_in_stmt(st, out); }
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

#[cfg(test)]
mod tests {
    use super::*;

    // `'atomic` fallback-chain inertness (see docs/qualifiers.md's `'atomic` section
    // and this file's `resolve_fallback` doc comment): `'actor` always precedes both
    // `'atomic` and `'guard` in the ordered chain, so whenever `{Actor, Guard, Atomic}`
    // (or any subset containing `Actor`) remain candidates simultaneously, `'actor`
    // wins the tie-break regardless of where `'atomic`/`'guard` sit relative to each
    // other. `'atomic` is reachable only via an explicit annotation or an explicit
    // call-site demand — never the plain fallback.
    #[test]
    fn atomic_never_wins_fallback_when_actor_present() {
        let candidates = [OwnerQual::Actor, OwnerQual::Guard, OwnerQual::Atomic];
        let result = resolve_fallback(&candidates, false, None, 256, true);
        assert_eq!(result, Some(OwnerQual::Actor), "expected 'actor to win the tie-break, not 'atomic or 'guard");
    }

    #[test]
    fn atomic_never_wins_fallback_actor_task_present() {
        let candidates = [OwnerQual::ActorTask, OwnerQual::Atomic, OwnerQual::Guard];
        let result = resolve_fallback(&candidates, false, None, 256, true);
        assert_eq!(result, Some(OwnerQual::ActorTask), "expected 'actor'task to win the tie-break over 'atomic");
    }

    // With no 'actor/'actor'task in the running, 'atomic DOES win over 'guard —
    // confirming it participates correctly in the chain once it's actually a
    // candidate (reachable only via explicit annotation/demand, per the doc above),
    // it isn't simply dead code that never resolves to anything.
    #[test]
    fn atomic_wins_over_guard_when_actor_absent() {
        let candidates = [OwnerQual::Guard, OwnerQual::Atomic];
        let result = resolve_fallback(&candidates, false, None, 256, true);
        assert_eq!(result, Some(OwnerQual::Atomic), "expected 'atomic to win over 'guard when 'actor isn't a candidate");
    }

    #[test]
    fn atomic_alone_resolves_to_atomic() {
        let candidates = [OwnerQual::Atomic];
        let result = resolve_fallback(&candidates, false, None, 256, true);
        assert_eq!(result, Some(OwnerQual::Atomic));
    }

    #[test]
    fn owned_still_wins_over_atomic() {
        let candidates = [OwnerQual::Owned, OwnerQual::Atomic, OwnerQual::Actor];
        let result = resolve_fallback(&candidates, false, None, 256, true);
        assert_eq!(result, Some(OwnerQual::Owned), "expected 'owned to still win the whole chain");
    }
}
