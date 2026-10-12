//! Borrow inference for bare-trait parameters of free functions (docs/qualifiers.md,
//! "Design: borrow inference for bare-trait parameters", phase 2).
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
    /// Names that occur as a bare value (not a call's callee), pipe target or method name.
    value_names: HashSet<String>,
    /// `(callee, argument shapes, any labeled/spread argument)` for every `f(..)` call.
    call_sites: Vec<(String, Vec<ArgShape>, bool)>,
}

impl Transpiler {
    /// Per-file part: gather the free functions and the usage facts of `program`.
    pub(crate) fn collect_trait_borrow_facts(&mut self, program: &Program) {
        let mut facts = std::mem::take(&mut self.trait_borrow_facts);
        collect_items(&program.items, &mut facts);
        self.trait_borrow_facts = facts;
    }

    /// Whole-graph part: run once after the outermost `deep_pre_scan`. Fills
    /// `trait_borrowed_fns` and rewrites the registered call signatures (`fn_sigs`).
    pub(crate) fn resolve_trait_borrow_params(&mut self) {
        let facts = std::mem::take(&mut self.trait_borrow_facts);
        let mut name_count: HashMap<&str, usize> = HashMap::new();
        for f in &facts.fns { *name_count.entry(f.name.as_str()).or_default() += 1; }
        let decls: HashMap<&str, &FnDecl> = facts.fns.iter()
            .filter(|f| name_count[f.name.as_str()] == 1)
            .map(|f| (f.name.as_str(), f))
            .collect();

        // Candidate (fn, param index) pairs.
        let mut borrowed: HashSet<(String, usize)> = HashSet::new();
        for f in decls.values() {
            if !self.fn_is_trait_borrow_candidate(f, &facts.value_names) { continue; }
            for (i, p) in f.params.iter().enumerate() {
                if self.param_trait_name(p).is_some() { borrowed.insert((f.name.clone(), i)); }
            }
        }

        // Call-site guards.
        for (callee, shapes, labeled) in &facts.call_sites {
            let Some(f) = decls.get(callee.as_str()) else { continue };
            for i in 0..f.params.len() {
                let bad = *labeled || match shapes.get(i) {
                    Some(ArgShape::Other) => true,
                    Some(ArgShape::Ctor(n)) => !self.is_known_value_ctor(n),
                    _ => false,
                };
                if bad { borrowed.remove(&(callee.clone(), i)); }
            }
            if shapes.len() > f.params.len() {
                for i in 0..f.params.len() { borrowed.remove(&(callee.clone(), i)); }
            }
        }

        // Shrinking fixpoint over the forwarding relation.
        loop {
            let mut changed = false;
            let current: Vec<(String, usize)> = borrowed.iter().cloned().collect();
            for (fname, i) in current {
                let f = decls[fname.as_str()];
                if !self.param_is_borrow_only(f, i, &borrowed, &decls) {
                    borrowed.remove(&(fname, i));
                    changed = true;
                }
            }
            if !changed { break; }
        }

        // Publish: per-function flags, and the signatures call sites read.
        for f in decls.values() {
            let flags: Vec<bool> = (0..f.params.len()).map(|i| borrowed.contains(&(f.name.clone(), i))).collect();
            if !flags.iter().any(|b| *b) { continue; }
            let keys = [f.name.clone(), crate::transpiler::helpers::mangle_overload_name(&f.name, &f.params)];
            for key in keys {
                let Some(sig) = self.fn_sigs.get_mut(&key) else { continue };
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
    pub(crate) fn with_inferred_trait_borrows(&self, f: &FnDecl) -> Option<FnDecl> {
        if f.qualifier.is_some() { return None; }
        let flags = self.trait_borrowed_fns.get(&f.name)?;
        let mut out = f.clone();
        let mut any = false;
        for (p, borrowed) in out.params.iter_mut().zip(flags) {
            if let (true, Some(Type::Named(n))) = (*borrowed, p.ty.as_ref()) {
                if self.trait_method_names.contains_key(n.as_str()) {
                    p.ty = p.ty.take().map(|t| Type::Qualified(Box::new(t), OwnerQual::Borrow));
                    any = true;
                }
            }
        }
        any.then_some(out)
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

    fn fn_is_trait_borrow_candidate(&self, f: &FnDecl, value_names: &HashSet<String>) -> bool {
        f.qualifier.is_none()
            && !f.is_native && !f.task && !f.stream
            && f.type_params.is_empty()
            && f.attrs.is_empty()
            && f.name != "main"
            && !value_names.contains(&f.name)
            && !self.overloaded_fn_names.contains(&f.name)
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
        decls: &HashMap<&str, &FnDecl>,
    ) -> bool {
        let p = &f.params[idx];
        let Some(trait_name) = self.param_trait_name(p) else { return false };
        let Some(methods) = self.trait_method_names.get(trait_name.as_str()) else { return false };
        let is_p = |e: &Expr| matches!(&e.kind, ExprKind::Var(v) if *v == p.name);
        let mut total = 0usize;
        let mut safe = 0usize;
        let mut escapes = false;
        walk_stmts(&f.body, &mut |node| {
            let WalkNode::Expr(e) = node else { return };
            match &e.kind {
                ExprKind::Var(v) if *v == p.name => total += 1,
                ExprKind::MethodCall(recv, m, _) if is_p(recv) && methods.contains(m.as_str()) => safe += 1,
                ExprKind::Call(callee, args) => {
                    if let ExprKind::Var(g) = &callee.kind {
                        for (i, a) in args.iter().enumerate() {
                            if a.label.is_none() && !a.spread && is_p(&a.value)
                                && callee_param_borrowed(g, i, borrowed, decls)
                            {
                                safe += 1;
                            }
                        }
                    }
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

/// Is the `i`-th parameter of free function `g` a borrow (explicit `Trait&` or inferred)?
fn callee_param_borrowed(
    g: &str,
    i: usize,
    borrowed: &HashSet<(String, usize)>,
    decls: &HashMap<&str, &FnDecl>,
) -> bool {
    if borrowed.contains(&(g.to_string(), i)) { return true; }
    matches!(
        decls.get(g).and_then(|d| d.params.get(i)).and_then(|p| p.ty.as_ref()),
        Some(Type::Qualified(_, OwnerQual::Borrow))
    )
}

fn collect_items(items: &[Item], facts: &mut TraitBorrowFacts) {
    for item in items {
        match item {
            Item::Fn(f) => {
                if f.qualifier.is_none() { facts.fns.push(f.clone()); }
                scan(&f.body, facts);
            }
            Item::Struct(s) => {
                for m in &s.methods { scan(&m.body, facts); }
                for i in &s.inits { scan(&i.body, facts); }
                for t in &s.type_methods { scan(&t.body, facts); }
                for c in &s.conversions { scan(&c.body, facts); }
                for st in &s.setters { scan(&st.body, facts); }
            }
            Item::Enum(e) => {
                for m in &e.methods { scan(&m.body, facts); }
                for t in &e.type_methods { scan(&t.body, facts); }
            }
            Item::Trait(t) => { for d in &t.defaults { scan(&d.body, facts); } }
            Item::Ext(x) => { for m in &x.methods { scan(&m.body, facts); } }
            Item::Kernel(k) => { for m in &k.methods { scan(&m.body, facts); } }
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
            ExprKind::Var(n) if !callees.contains(&(e as *const Expr)) => {
                facts.value_names.insert(n.clone());
            }
            ExprKind::Pipe(_, name, _) | ExprKind::MethodCall(_, name, _)
            | ExprKind::OptionalMethodCall(_, name, _) => {
                facts.value_names.insert(name.clone());
            }
            _ => {}
        }
    });
}
