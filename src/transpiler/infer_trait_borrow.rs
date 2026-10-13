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
//! Trait contracts (phase 4). A bare-trait parameter of a trait method signature or default
//! method is *borrowed* by default (`&dyn Trait`) unless the author opts into by-value with
//! `Trait'owned p`; every implementer (struct, enum, `ext`) and every call through a trait
//! object or generic bound follows the contract, never the body. An implementer or default
//! method whose body consumes such a parameter is a diagnostic naming the `'owned` fix.
//!
//! A `mut` bare-trait parameter is lent (`&mut dyn Trait`) unconditionally, like a `mut`
//! struct parameter (docs/book.md "`mut` vs `var` on a struct parameter"): the callee mutates
//! the caller's value, so the signature never depends on the body, and a body that consumes it
//! is a diagnostic. A `var` parameter is already an out-parameter (`&mut Box<dyn Trait>`).
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
    /// Types that may implement traits: `(type name, declared trait names, methods)` for every
    /// struct, enum and `ext` of the graph.
    implementers: Vec<(String, Vec<String>, Vec<FnDecl>)>,
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

/// `ty` borrowed the way parameter `p` is: `&dyn Trait`, or `&mut dyn Trait` for a `mut` one.
fn borrowed_ty(p: &Param, ty: &Type) -> Type {
    let q = if p.mutable { OwnerQual::BorrowMut } else { OwnerQual::Borrow };
    Type::Qualified(Box::new(ty.clone()), q)
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
                // A `mut` parameter is lent whatever its call sites look like.
                if bad && !f.params[i].mutable { borrowed.remove(&(key.to_string(), i)); }
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

        // Trait contracts: a bare-trait param of a trait method is borrowed unless `'owned`.
        // A name declared by exactly one trait resolves a forwarded call to its contract.
        let mut contract_trait_of: HashMap<&str, Vec<&str>> = HashMap::new();
        for (t, names) in &self.trait_method_names {
            for m in names { contract_trait_of.entry(m.as_str()).or_default().push(t.as_str()); }
        }
        let contract_map: HashMap<String, Vec<bool>> = self.trait_method_params.iter()
            .map(|(k, ps)| (k.clone(), ps.iter().map(|p| self.param_trait_name(p).is_some()).collect()))
            .collect();
        let contract_flags = |t: &str, m: &str| -> Vec<bool> {
            contract_map.get(&format!("{}::{}", t, m)).cloned().unwrap_or_default()
        };
        let mut contract_keys: Vec<(String, String, Vec<bool>)> = Vec::new(); // (key, method, flags)
        for (m, ts) in &contract_trait_of {
            if ts.len() != 1 { continue; }
            let flags = contract_flags(ts[0], m);
            if !flags.iter().any(|b| *b) { continue; }
            let key = format!("@{}::{}", ts[0], m);
            for (i, b) in flags.iter().enumerate() { if *b { borrowed.insert((key.clone(), i)); } }
            method_key_by_name.insert(m, key.clone());
            contract_keys.push((key, m.to_string(), flags));
        }

        // Shrinking fixpoint over the forwarding relation.
        loop {
            let mut changed = false;
            let current: Vec<(String, usize)> = borrowed.iter().cloned().collect();
            for (key, i) in current {
                if key.starts_with('@') { continue; } // contract entries are fixed
                let f = decls[key.as_str()];
                if f.params[i].mutable { continue; } // lent unconditionally, checked below
                if !self.param_is_borrow_only(f, i, &borrowed, &decls, &method_key_by_name) {
                    borrowed.remove(&(key, i));
                    changed = true;
                }
            }
            if !changed { break; }
        }

        // A lent `mut` parameter whose body consumes it has no valid signature.
        for (key, i) in &borrowed {
            if key.starts_with('@') { continue; }
            let f = decls[key.as_str()];
            let p = &f.params[*i];
            if p.mutable && !self.param_is_borrow_only(f, *i, &borrowed, &decls, &method_key_by_name) {
                self.push_error(p.line, p.col, format!(
                    "`{}` is a `mut` trait parameter, which is lent (`&mut dyn {}`), but `{}` consumes it \
                     (stores, returns, captures or passes it by value); declare it `{}'owned` to take ownership",
                    p.name, self.param_trait_name(p).unwrap_or_default(), f.name,
                    self.param_trait_name(p).unwrap_or_default()));
            }
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
                        p.ty.as_ref().map(|t| borrowed_ty(p, t))
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
                        *slot = borrowed_ty(p, ty);
                    }
                }
            }
            self.trait_borrowed_fns.insert(f.name.clone(), flags);
        }

        // Contract call-by-name entries (trait/dyn receivers are handled from the trait's own
        // parameter list at the call site; these serve receivers of unknown type).
        for (key, m, flags) in &contract_keys {
            let (_, tname_m) = key[1..].split_once("::").unwrap();
            let t = &key[1..key.len() - tname_m.len() - 2];
            let types: Vec<Option<Type>> = self.trait_method_params.get(&format!("{}::{}", t, m))
                .map(|ps| ps.iter().zip(flags).map(|(p, b)| if *b {
                    p.ty.as_ref().map(|ty| borrowed_ty(p, ty))
                } else { None }).collect())
                .unwrap_or_default();
            self.trait_borrowed_method_by_name.insert(m.clone(), key.clone());
            self.trait_borrowed_methods.insert(key.clone(), types);
        }

        // Implementers follow the contract of every trait they implement, explicitly
        // (`as Trait`) or by having all of its required methods.
        for (ty, protos, methods) in &facts.implementers {
            let names: HashSet<&str> = methods.iter().map(|m| m.name.as_str()).collect();
            let implemented: Vec<String> = self.trait_method_names.iter()
                .filter(|(t, req)| protos.contains(t)
                    || (!req.is_empty() && req.iter().all(|r| names.contains(r.as_str()))))
                .map(|(t, _)| t.clone())
                .collect();
            for m in methods {
                let mut flags: Option<Vec<bool>> = None;
                let mut trait_of = String::new();
                for t in &implemented {
                    if !self.trait_method_names[t.as_str()].contains(&m.name) { continue; }
                    let f = contract_flags(t, &m.name);
                    if !f.iter().any(|b| *b) { continue; }
                    match &flags {
                        None => { flags = Some(f); trait_of = t.clone(); }
                        Some(prev) if *prev == f => {}
                        Some(_) => self.push_error(m.line, m.col, format!(
                            "`{}.{}` implements methods of several traits whose bare-trait parameters \
                             are borrowed differently; declare the by-value ones `Trait'owned`",
                            ty, m.name)),
                    }
                }
                let Some(flags) = flags else { continue };
                let mut types: Vec<Option<Type>> = vec![None; m.params.len()];
                for (i, p) in m.params.iter().enumerate() {
                    if !flags.get(i).copied().unwrap_or(false) { continue; }
                    let trait_mut = self.trait_method_params.get(&format!("{}::{}", trait_of, m.name))
                        .and_then(|ps| ps.get(i)).map(|tp| tp.mutable);
                    if trait_mut.is_some_and(|tm| tm != p.mutable) {
                        self.push_error(p.line, p.col, format!(
                            "parameter `{}` of `{}.{}` must be {}`mut` like trait `{}`'s parameter",
                            p.name, ty, m.name, if p.mutable { "not " } else { "" }, trait_of));
                        continue;
                    }
                    match p.ty.as_ref() {
                        Some(t @ Type::Named(n)) if self.trait_method_names.contains_key(n.as_str()) && self.param_trait_name(p).is_some() => {
                            if !self.trait_param_only_borrowed(m, i, &borrowed, &decls, &method_key_by_name) {
                                self.push_error(p.line, p.col, format!(
                                    "`{}` is consumed in `{}.{}`, but trait `{}` passes its bare-trait parameter by reference; \
                                     declare it `{}'owned {}` in the trait and here to take ownership",
                                    p.name, ty, m.name, trait_of, n, p.name));
                            }
                            types[i] = Some(borrowed_ty(p, t));
                        }
                        _ => self.push_error(p.line, p.col, format!(
                            "parameter `{}` of `{}.{}` must be a bare trait type to match trait `{}`, whose contract passes it by reference \
                             (or declare it `'owned` in the trait too)",
                            p.name, ty, m.name, trait_of)),
                    }
                }
                self.trait_borrowed_methods.insert(format!("{}::{}", ty, m.name), types);
            }
        }

        // Explicit `T&` / `mut T&` parameters of enum and `ext` methods. A struct's methods
        // are registered with their declaration, but an enum's are not, so a call site has no
        // parameter list to read the borrow from: record them in the same table.
        for (ty, _, methods) in &facts.implementers {
            for m in methods {
                if !m.params.iter().any(|p| matches!(p.ty, Some(Type::Qualified(_, OwnerQual::Borrow | OwnerQual::BorrowMut)))) {
                    continue;
                }
                let entry = self.trait_borrowed_methods
                    .entry(format!("{}::{}", ty, m.name))
                    .or_insert_with(|| vec![None; m.params.len()]);
                for (i, p) in m.params.iter().enumerate() {
                    if let (Some(None), Some(t @ Type::Qualified(_, OwnerQual::Borrow | OwnerQual::BorrowMut))) = (entry.get(i), p.ty.as_ref()) {
                        entry[i] = Some(t.clone());
                    }
                }
            }
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
                    let q = if p.mutable { OwnerQual::BorrowMut } else { OwnerQual::Borrow };
                    p.ty = p.ty.take().map(|t| Type::Qualified(Box::new(t), q));
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

    /// `params` of a trait method (signature or default body) with every contract-borrowed
    /// bare-trait parameter rewritten to `Trait&`; `None` when nothing changes.
    pub(crate) fn contract_borrowed_params(&self, params: &[Param]) -> Option<Vec<Param>> {
        if !params.iter().any(|p| self.param_trait_name(p).is_some()) { return None; }
        Some(params.iter().map(|p| {
            let mut q = p.clone();
            if self.param_trait_name(p).is_some() {
                q.ty = q.ty.take().map(|t| borrowed_ty(p, &t));
            }
            q
        }).collect())
    }

    pub(crate) fn param_trait_name(&self, p: &Param) -> Option<String> {
        if p.rebindable || p.var_mut || p.owned || p.variadic || p.default.is_some() {
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

    fn trait_param_only_borrowed(
        &self,
        f: &FnDecl,
        idx: usize,
        borrowed: &HashSet<(String, usize)>,
        decls: &HashMap<String, &FnDecl>,
        method_key_by_name: &HashMap<&str, String>,
    ) -> bool {
        self.param_is_borrow_only(f, idx, borrowed, decls, method_key_by_name)
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
        Some(Type::Qualified(_, OwnerQual::Borrow | OwnerQual::BorrowMut))
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
                facts.implementers.push((s.name.clone(), s.protocols.clone(), s.methods.clone()));
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
                facts.implementers.push((e.name.clone(), e.protocols.clone(), e.methods.clone()));
                count_methods(&e.methods, facts);
                for m in &e.methods { scan(&m.body, facts); }
                for t in &e.type_methods { scan(&t.body, facts); }
            }
            Item::Trait(t) => {
                count_methods(&t.defaults, facts);
                for d in &t.defaults { scan(&d.body, facts); }
            }
            Item::Ext(x) => {
                facts.implementers.push((x.type_name.clone(), x.traits.clone(), x.methods.clone()));
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
