//! Borrow inference for bare-trait parameters of free functions and struct methods
//! (docs/qualifiers.md, "Design: borrow inference for bare-trait parameters", phases 2–3).
//!
//! A bare trait name in parameter position is dynamic dispatch (`Box<dyn Trait>`, by value).
//! A parameter that the body never consumes is rewritten, as if the author had written
//! `Trait& p`, to `&dyn Trait`; phase 1 already knows how to emit and call that.
//!
//! The decision is a property of the callee's declaration alone: a parameter is *borrowed*
//! when every occurrence of it in the body is a trait-method receiver (`p.m(..)`) or an
//! unlabeled argument handed to a borrowed parameter of another free function. Forwarding
//! makes this recursive, so it runs as a shrinking fixpoint over every free function of the
//! whole reachable file graph (`deep_pre_scan` gathers the facts, `resolve_trait_borrow_params`
//! decides once the outermost scan is complete). Anything else — stored, returned, assigned,
//! captured by a closure or task, passed to an unresolved callee, `as`-cast, interpolated —
//! keeps the by-value `Box<dyn Trait>`.
//!
//! Struct methods (phase 3) use the same analysis, keyed `Type::method`. A call site may not
//! know its receiver's type (dyn receiver, untyped closure parameter), so a method is only a
//! candidate when its *name* identifies it program-wide: declared by exactly one method of any
//! kind, by no trait (an implementer must match its trait's signature), and not also a free
//! function, field, piped or bare-value name. Call sites then resolve by receiver type when
//! known and by name otherwise.
//!
//! A function is left alone entirely when its signature could be observed through a path that
//! does not go through the borrow-aware call emission: used as a value (`map(f)`), piped
//! (`x |> f`), named like any method called anywhere, overloaded, generic, `task`/`stream`,
//! attributed (`@provide`, ...), or called with labeled/spread arguments. A call site whose
//! argument for a borrowed position is not a variable, a field, or a constructor call (e.g.
//! `f(make())`, whose `Box<dyn Trait>` result would need `&*`) also keeps the parameter by value.

use super::Transpiler;
use crate::ast::*;
use std::collections::{HashMap, HashSet};

/// Syntactic shape of one call argument, resolved against the type tables only once every
/// file of the graph has been scanned.
#[derive(Clone)]
pub(crate) enum ArgShape {
    Var,
    Field,
    /// `Name(..)`: fresh value if `Name` turns out to be a struct/enum constructor.
    Ctor(String),
    Other,
}

#[derive(Default)]
pub(crate) struct TraitBorrowFacts {
    /// Every free function (`def f(..)`, no `Type.` qualifier) of the file graph.
    fns: Vec<FnDecl>,
    /// Methods of non-generic structs: `(struct name, decl)`.
    methods: Vec<(String, FnDecl)>,
    /// How many methods of any kind (struct, enum, ext, kernel, trait default) bear each name.
    method_name_count: HashMap<String, usize>,
    /// Names that occur as a bare value (not a call's callee) or pipe target.
    value_names: HashSet<String>,
    /// Names of methods called anywhere (`x.m(..)`).
    called_method_names: HashSet<String>,
    /// Names read as a field anywhere (`x.name`).
    field_names: HashSet<String>,
    /// `(callee, argument shapes, any labeled/spread argument)` for every `f(..)` call.
    call_sites: Vec<(String, Vec<ArgShape>, bool)>,
    /// Same for every `x.m(..)` call, keyed by method name.
    method_call_sites: Vec<(String, Vec<ArgShape>, bool)>,
}

impl Transpiler {
    /// Per-file part: gather the free functions, methods and usage facts of `program`.
    pub(crate) fn collect_trait_borrow_facts(&mut self, program: &Program) {
        let mut facts = std::mem::take(&mut self.trait_borrow_facts);
        collect_items(&program.items, &mut facts);
        self.trait_borrow_facts = facts;
    }

    /// Whole-graph part: run once after the outermost `deep_pre_scan`. Fills
    /// `trait_borrowed_fns` / `trait_borrowed_methods` and rewrites the registered call
    /// signatures of free functions (`fn_sigs`).
    pub(crate) fn resolve_trait_borrow_params(&mut self) {
        let facts = std::mem::take(&mut self.trait_borrow_facts);

        // Callee declarations by key: a free function is `name`, a method `Type::name`.
        let mut free_count: HashMap<&str, usize> = HashMap::new();
        for f in &facts.fns { *free_count.entry(f.name.as_str()).or_default() += 1; }
        let mut decls: HashMap<String, &FnDecl> = HashMap::new();
        for f in facts.fns.iter().filter(|f| free_count[f.name.as_str()] == 1) {
            decls.insert(f.name.clone(), f);
        }
        let all_trait_methods: HashSet<&str> = self.trait_method_names.values()
            .flat_map(|ms| ms.iter().map(|m| m.as_str()))
            .collect();
        // Methods whose bare name identifies them program-wide (see the module doc).
        let mut method_key_by_name: HashMap<&str, String> = HashMap::new();
        for (ty, m) in &facts.methods {
            let name = m.name.as_str();
            let unique = facts.method_name_count.get(name) == Some(&1)
                && !all_trait_methods.contains(name)
                && !free_count.contains_key(name)
                && !facts.value_names.contains(name)
                && !facts.field_names.contains(name);
            if unique {
                let key = format!("{}::{}", ty, name);
                decls.insert(key.clone(), m);
                method_key_by_name.insert(name, key);
            }
        }

        // Candidate (callee, param index) pairs.
        let mut borrowed: HashSet<(String, usize)> = HashSet::new();
        for (key, f) in &decls {
            let is_method = key.contains("::");
            let ok = if is_method {
                self.method_is_trait_borrow_candidate(f)
            } else {
                self.fn_is_trait_borrow_candidate(f, &facts.value_names, &facts.called_method_names)
            };
            if !ok { continue; }
            for (i, p) in f.params.iter().enumerate() {
                if self.param_trait_name(p).is_some() { borrowed.insert((key.clone(), i)); }
            }
        }

        // Call-site guards, for `f(..)` calls and `x.m(..)` calls alike.
        let guard = |borrowed: &mut HashSet<(String, usize)>, key: &str, shapes: &[ArgShape], labeled: bool| {
            let Some(f) = decls.get(key) else { return };
            for i in 0..f.params.len() {
                let bad = labeled || shapes.len() > f.params.len() || match shapes.get(i) {
                    Some(ArgShape::Other) => true,
                    Some(ArgShape::Ctor(n)) => !self.is_known_value_ctor(n),
                    _ => false,
                };
                if bad { borrowed.remove(&(key.to_string(), i)); }
            }
        };
        for (callee, shapes, labeled) in &facts.call_sites {
            guard(&mut borrowed, callee, shapes, *labeled);
        }
        for (name, shapes, labeled) in &facts.method_call_sites {
            if let Some(key) = method_key_by_name.get(name.as_str()) {
                guard(&mut borrowed, key, shapes, *labeled);
            }
        }

        // Shrinking fixpoint over the forwarding relation.
        loop {
            let mut changed = false;
            let current: Vec<(String, usize)> = borrowed.iter().cloned().collect();
            for (key, i) in current {
                let f = decls[key.as_str()];
                if !self.param_is_borrow_only(f, i, &borrowed, &decls, &method_key_by_name) {
                    borrowed.remove(&(key, i));
                    changed = true;
                }
            }
            if !changed { break; }
        }

        // Publish.
        for (key, f) in &decls {
            let flags: Vec<bool> = (0..f.params.len()).map(|i| borrowed.contains(&(key.clone(), i))).collect();
            if !flags.iter().any(|b| *b) { continue; }
            if key.contains("::") {
                // Methods: a table of replacement parameter types the call sites consult
                // (the registered declarations stay as written, so re-registration of a
                // re-scanned file still recognises them).
                let types: Vec<Option<Type>> = f.params.iter().zip(&flags)
                    .map(|(p, b)| if *b {
                        p.ty.as_ref().map(|t| Type::Qualified(Box::new(t.clone()), OwnerQual::Borrow))
                    } else { None })
                    .collect();
                self.trait_borrowed_method_by_name.insert(f.name.clone(), key.clone());
                self.trait_borrowed_methods.insert(key.clone(), types);
                continue;
            }
            let sig_keys = [f.name.clone(), crate::transpiler::helpers::mangle_overload_name(&f.name, &f.params)];
            for sk in sig_keys {
                let Some(sig) = self.fn_sigs.get_mut(&sk) else { continue };
                for (i, p) in f.params.iter().enumerate() {
                    if !flags[i] { continue; }
                    // `fn_sigs` skips untyped params, so the index is among the typed ones.
                    let sig_idx = f.params[..i].iter().filter(|q| q.ty.is_some()).count();
                    if let (Some(slot), Some(ty)) = (sig.get_mut(sig_idx), p.ty.as_ref()) {
                        *slot = Type::Qualified(Box::new(ty.clone()), OwnerQual::Borrow);
                    }
                }
            }
            self.trait_borrowed_fns.insert(f.name.clone(), flags);
        }
        self.trait_borrow_facts = TraitBorrowFacts::default();
    }

    /// `f` with its borrowed bare-trait params rewritten to `Trait&`, when inference chose any.
    /// `self_ty` is the owning struct for a method.
    pub(crate) fn with_inferred_trait_borrows(&self, f: &FnDecl, self_ty: Option<&str>) -> Option<FnDecl> {
        if f.qualifier.is_some() { return None; }
        let flags: Vec<bool> = match self_ty {
            None => self.trait_borrowed_fns.get(&f.name)?.clone(),
            Some(t) => self.trait_borrowed_methods.get(&format!("{}::{}", t, f.name))?
                .iter().map(|o| o.is_some()).collect(),
        };
        let mut out = f.clone();
        let mut any = false;
        for (p, borrowed) in out.params.iter_mut().zip(&flags) {
            if let (true, Some(Type::Named(n))) = (*borrowed, p.ty.as_ref()) {
                if self.trait_method_names.contains_key(n.as_str()) {
                    p.ty = p.ty.take().map(|t| Type::Qualified(Box::new(t), OwnerQual::Borrow));
                    any = true;
                }
            }
        }
        any.then_some(out)
    }

    /// The inferred-borrow replacement type of parameter `i` of `struct_name::method`, if any.
    pub(crate) fn inferred_method_param_type(&self, struct_name: &str, method: &str, i: usize) -> Option<&Type> {
        self.trait_borrowed_methods.get(&format!("{}::{}", struct_name, method))?
            .get(i)?.as_ref()
    }

    /// Replacement parameter types for a call to `method` whose receiver type is unknown at
    /// the call site: resolved by the (program-wide unique) method name.
    pub(crate) fn inferred_method_param_types_by_name(&self, method: &str) -> Option<&Vec<Option<Type>>> {
        let key = self.trait_borrowed_method_by_name.get(method)?;
        self.trait_borrowed_methods.get(key)
    }

    fn param_trait_name(&self, p: &Param) -> Option<String> {
        if p.mutable || p.rebindable || p.var_mut || p.owned || p.variadic || p.default.is_some() {
            return None;
        }
        match p.ty.as_ref()? {
            Type::Named(n) if self.trait_method_names.contains_key(n.as_str()) => Some(n.clone()),
            _ => None,
        }
    }

    fn fn_is_trait_borrow_candidate(
        &self,
        f: &FnDecl,
        value_names: &HashSet<String>,
        called_method_names: &HashSet<String>,
    ) -> bool {
        f.qualifier.is_none()
            && !f.is_native && !f.task && !f.stream
            && f.type_params.is_empty()
            && f.attrs.is_empty()
            && f.name != "main"
            && !value_names.contains(&f.name)
            && !called_method_names.contains(&f.name)
            && !self.overloaded_fn_names.contains(&f.name)
    }

    fn method_is_trait_borrow_candidate(&self, f: &FnDecl) -> bool {
        !f.is_native && !f.task && !f.stream
            && f.type_params.is_empty()
            && f.attrs.is_empty()
    }

    /// `Name(..)` builds a fresh value: a struct/enum constructor, not a function returning one.
    fn is_known_value_ctor(&self, name: &str) -> bool {
        self.struct_fields.contains_key(name)
            || self.type_sizes.contains_key(name)
            || self.enum_variants.contains_key(name)
    }

    /// True when every occurrence of `f.params[idx]` in the body is a borrow use.
    fn param_is_borrow_only(
        &self,
        f: &FnDecl,
        idx: usize,
        borrowed: &HashSet<(String, usize)>,
        decls: &HashMap<String, &FnDecl>,
        method_key_by_name: &HashMap<&str, String>,
    ) -> bool {
        let p = &f.params[idx];
        let Some(trait_name) = self.param_trait_name(p) else { return false };
        let Some(methods) = self.trait_method_names.get(trait_name.as_str()) else { return false };
        let is_p = |e: &Expr| matches!(&e.kind, ExprKind::Var(v) if *v == p.name);
        let forwarded = |key: &str, args: &[Arg]| -> usize {
            args.iter().enumerate()
                .filter(|(i, a)| a.label.is_none() && !a.spread && is_p(&a.value)
                    && callee_param_borrowed(key, *i, borrowed, decls))
                .count()
        };
        let mut total = 0usize;
        let mut safe = 0usize;
        let mut escapes = false;
        walk_stmts(&f.body, &mut |node| {
            let WalkNode::Expr(e) = node else { return };
            match &e.kind {
                ExprKind::Var(v) if *v == p.name => total += 1,
                ExprKind::MethodCall(recv, m, args) => {
                    if is_p(recv) && methods.contains(m.as_str()) { safe += 1; }
                    if let Some(key) = method_key_by_name.get(m.as_str()) {
                        safe += forwarded(key, args);
                    }
                }
                ExprKind::Call(callee, args) => {
                    if let ExprKind::Var(g) = &callee.kind { safe += forwarded(g, args); }
                }
                // Captured by a closure / task: it outlives the call.
                ExprKind::Closure(..) | ExprKind::Task(_) => {
                    let mut inside = 0usize;
                    walk_expr(e, &mut |n| {
                        if let WalkNode::Expr(x) = n { if is_p(x) { inside += 1; } }
                    });
                    if inside > 0 { escapes = true; }
                }
                _ => {}
            }
        });
        total == safe && !escapes
    }
}

/// Is parameter `i` of callee `key` a borrow (explicit `Trait&` or inferred)?
fn callee_param_borrowed(
    key: &str,
    i: usize,
    borrowed: &HashSet<(String, usize)>,
    decls: &HashMap<String, &FnDecl>,
) -> bool {
    if borrowed.contains(&(key.to_string(), i)) { return true; }
    matches!(
        decls.get(key).and_then(|d| d.params.get(i)).and_then(|p| p.ty.as_ref()),
        Some(Type::Qualified(_, OwnerQual::Borrow))
    )
}

fn count_methods(ms: &[FnDecl], facts: &mut TraitBorrowFacts) {
    for m in ms { *facts.method_name_count.entry(m.name.clone()).or_default() += 1; }
}

fn collect_items(items: &[Item], facts: &mut TraitBorrowFacts) {
    for item in items {
        match item {
            Item::Fn(f) => {
                if f.qualifier.is_none() { facts.fns.push(f.clone()); }
                scan(&f.body, facts);
            }
            Item::Struct(s) => {
                count_methods(&s.methods, facts);
                if s.type_params.is_empty() {
                    for m in &s.methods { facts.methods.push((s.name.clone(), m.clone())); }
                }
                for m in &s.methods { scan(&m.body, facts); }
                for i in &s.inits { scan(&i.body, facts); }
                for t in &s.type_methods { scan(&t.body, facts); }
                for c in &s.conversions { scan(&c.body, facts); }
                for st in &s.setters { scan(&st.body, facts); }
            }
            Item::Enum(e) => {
                count_methods(&e.methods, facts);
                for m in &e.methods { scan(&m.body, facts); }
                for t in &e.type_methods { scan(&t.body, facts); }
            }
            Item::Trait(t) => {
                count_methods(&t.defaults, facts);
                for d in &t.defaults { scan(&d.body, facts); }
            }
            Item::Ext(x) => {
                count_methods(&x.methods, facts);
                for m in &x.methods { scan(&m.body, facts); }
            }
            Item::Kernel(k) => {
                count_methods(&k.methods, facts);
                for m in &k.methods { scan(&m.body, facts); }
            }
            Item::Mod(m) => collect_items(&m.items, facts),
            Item::Let(l) => scan(&[Stmt::Let(l.clone())], facts),
            Item::Stmt(s) => scan(std::slice::from_ref(s), facts),
            Item::Use(_) | Item::Alias(_) => {}
        }
    }
}

fn arg_shape(e: &Expr) -> ArgShape {
    match &e.kind {
        ExprKind::Var(_) => ArgShape::Var,
        ExprKind::Field(..) => ArgShape::Field,
        ExprKind::Call(c, _) => match &c.kind {
            ExprKind::Var(n) => ArgShape::Ctor(n.clone()),
            _ => ArgShape::Other,
        },
        _ => ArgShape::Other,
    }
}

fn scan(stmts: &[Stmt], facts: &mut TraitBorrowFacts) {
    // Pre-order: a call is visited before its callee, so the callee expression can be set
    // aside and not mistaken for a bare value use of the function's name.
    let mut callees: HashSet<*const Expr> = HashSet::new();
    walk_stmts(stmts, &mut |node| {
        let WalkNode::Expr(e) = node else { return };
        match &e.kind {
            ExprKind::Call(callee, args) => {
                if let ExprKind::Var(f) = &callee.kind {
                    callees.insert(&**callee as *const Expr);
                    let labeled = args.iter().any(|a| a.label.is_some() || a.spread);
                    let shapes = args.iter().map(|a| arg_shape(&a.value)).collect();
                    facts.call_sites.push((f.clone(), shapes, labeled));
                }
            }
            ExprKind::MethodCall(_, name, args) => {
                facts.called_method_names.insert(name.clone());
                let labeled = args.iter().any(|a| a.label.is_some() || a.spread);
                let shapes = args.iter().map(|a| arg_shape(&a.value)).collect();
                facts.method_call_sites.push((name.clone(), shapes, labeled));
            }
            ExprKind::OptionalMethodCall(_, name, _) => {
                facts.called_method_names.insert(name.clone());
                facts.value_names.insert(name.clone());
            }
            ExprKind::Var(n) if !callees.contains(&(e as *const Expr)) => {
                facts.value_names.insert(n.clone());
            }
            ExprKind::Pipe(_, name, _) => { facts.value_names.insert(name.clone()); }
            ExprKind::Field(_, name) | ExprKind::OptionalField(_, name) => {
                facts.field_names.insert(name.clone());
            }
            _ => {}
        }
    });
}
