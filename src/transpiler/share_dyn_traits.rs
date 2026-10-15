//! Reference-counted (`'shared`) representation for non-mutating trait objects held by structs.
//!
//! A bare trait name in type position is `Box<dyn Trait>`, which is not `Clone`: a struct holding
//! one (directly, as `Trait?`, `[Trait]`, or transitively through another struct) cannot derive
//! `Clone`/`PartialEq`, so Boring's by-value copy semantics (`let b = a`, `kids[i]`,
//! `node.children`, passing it to a function) have no valid Rust. Borrowing every use
//! (`borrow_noclone_struct_params`) covers parameters, but not values that are genuinely copied
//! — a tree of such structs rebuilt on every update pass, say.
//!
//! A trait whose methods are all `req` (`&self`) can be shared instead: this pass rewrites every
//! type position naming such a trait, program-wide, to `Trait'shared` (`Arc<dyn Trait>` /
//! `Rc<dyn Trait>`, exactly as if the author had written the qualifier). A shared pointer is
//! `Clone` whatever it points at, so the holding structs derive `Clone` again, and the existing
//! `'shared` machinery (implicit alias on assignment, `Arc::new` coercion at construction sites,
//! no `.clone()` ceremony) handles the rest. It cannot lose a mutation, since a `req`-only trait
//! has none to perform.
//!
//! The decision is per *trait*, never per use, so every position agrees on one representation
//! (an `Arc<dyn T>` can never be turned back into a `Box<dyn T>`):
//!   - the trait is non-generic, has no associated types, and every method — its own, its
//!     defaults, and those of its user-declared supertraits — is non-mutating; and
//!   - some struct field type names it bare (through `?`, `[T]`, `{K=V}`, `{T}`, tuples).
//!
//! Traits that have a `def` method keep `Box<dyn Trait>` (and the borrow/no-`Clone` handling of
//! commit 299fbf5b).
//!
//! The facts come from a cheap collection pass over the whole reachable file graph
//! (`Transpiler::dyn_fact_pass`, run by `deep_pre_scan` before the real scan); the rewrite is
//! then applied to each program before anything registers its types, so every table agrees.

use std::collections::{HashMap, HashSet};

use crate::ast::*;
use crate::ast_walk::Walker;

/// What the collection pass learned about one trait.
#[derive(Default, Clone)]
pub(crate) struct TraitFact {
    pub(crate) all_req: bool,
    pub(crate) parents: Vec<String>,
}

#[derive(Default)]
pub(crate) struct DynFacts {
    pub(crate) traits: HashMap<String, TraitFact>,
    /// Every bare name appearing in a struct field's type, through containers. Whether a name is
    /// a trait is only known once all files are scanned, hence the late intersection.
    pub(crate) held: HashSet<String>,
}

impl DynFacts {
    pub(crate) fn collect(&mut self, program: &Program) {
        self.collect_items(&program.items);
    }

    fn collect_items(&mut self, items: &[Item]) {
        for item in items {
            match item {
                Item::Trait(t) => {
                    let all_req = t.type_params.is_empty()
                        && t.assoc_types.is_empty()
                        && t.type_signatures.is_empty()
                        && t.signatures.iter().all(|s| !s.mutating && !s.task && !s.stream && s.type_params.is_empty())
                        && t.defaults.iter().all(|d| !d.mutating && !d.task && !d.stream && d.type_params.is_empty());
                    self.traits.insert(t.name.clone(), TraitFact { all_req, parents: t.parents.clone() });
                }
                Item::Struct(s) => {
                    for f in &s.fields { Self::held_names(&f.ty, &mut self.held); }
                }
                Item::Mod(m) => self.collect_items(&m.items),
                _ => {}
            }
        }
    }

    fn held_names(ty: &Type, out: &mut HashSet<String>) {
        match ty {
            Type::Named(n) => { out.insert(n.clone()); }
            Type::Dyn(i) | Type::Optional(i) | Type::Array(i) | Type::ArrayN(i, _) | Type::Set(i) | Type::Mut(i) => {
                Self::held_names(i, out)
            }
            Type::Dict(k, v) => { Self::held_names(k, out); Self::held_names(v, out); }
            Type::Tuple(es) => for e in es { Self::held_names(e, out); },
            _ => {}
        }
    }

    /// The traits to share: all-`req` (transitively) and held by some struct.
    pub(crate) fn targets(&self) -> HashSet<String> {
        fn all_req(f: &DynFacts, name: &str, seen: &mut HashSet<String>) -> bool {
            if !seen.insert(name.to_string()) { return true; }
            match f.traits.get(name) {
                // Not a user trait (`Debug`, `Display`, …): says nothing about mutation.
                None => true,
                Some(t) => t.all_req && t.parents.iter().all(|p| all_req(f, p, seen)),
            }
        }
        self.held.iter()
            .filter(|n| self.traits.contains_key(n.as_str()) && all_req(self, n, &mut HashSet::new()))
            .cloned()
            .collect()
    }
}

/// Rewrites every type in `program` naming a target trait to `Trait'shared`.
///
/// A parameter written as the bare trait is, to borrow inference (`infer_trait_borrow.rs`),
/// either borrowed (`&dyn Trait`) or by value. So it is left bare while that decision is
/// pending (`borrow` is `None`) and afterwards when it was decided borrowed — or is a trait
/// contract parameter, always borrowed; a by-value one becomes `Trait'shared` like any other
/// position.
pub(crate) fn rewrite_program(program: &mut Program, targets: &HashSet<String>, borrow: Option<&BorrowView>) {
    rw_items(&mut program.items, targets, &borrow);
}

/// The outcome of borrow inference, as published by `resolve_trait_borrow_params`.
pub(crate) struct BorrowView<'a> {
    pub(crate) fns: &'a HashMap<String, Vec<bool>>,
    pub(crate) methods: &'a HashMap<String, Vec<Option<Type>>>,
    pub(crate) by_name: &'a HashMap<String, String>,
}

/// `ty` as a by-value parameter of a program where `targets` are shared: a bare target trait
/// becomes `Trait'shared`, anything else is unchanged.
pub(crate) fn shared_if_bare(ty: &Type, targets: &HashSet<String>) -> Type {
    match ty {
        Type::Named(n) if targets.contains(n.as_str()) =>
            Type::Qualified(Box::new(Type::Named(n.clone())), OwnerQual::Shared),
        _ => ty.clone(),
    }
}

type B<'a, 'b> = &'b Option<&'a BorrowView<'a>>;

fn is_target_bare(p: &Param, t: &HashSet<String>) -> bool {
    matches!(p.ty.as_ref().map(Type::without_mut), Some(Type::Named(n)) if t.contains(n.as_str()))
}

/// Should bare parameter `i` of the function/method `key` (`Type::name` for a method) stay bare?
fn keeps_bare(b: B, key: &str, name: &str, i: usize, is_method: bool) -> bool {
    let Some(v) = b else { return true };
    if is_method {
        if v.methods.get(key).is_some_and(|ts| ts.get(i).is_some_and(Option::is_some)) { return true; }
        // An implementer's method follows its trait's contract (`@Trait::name`).
        return v.by_name.get(name).is_some_and(|k| k.starts_with('@')
            && v.methods.get(k).is_some_and(|ts| ts.get(i).is_some_and(Option::is_some)));
    }
    v.fns.get(key).is_some_and(|fl| fl.get(i).copied().unwrap_or(false))
}

/// `rw_fn`, except the bare parameters that `keeps_bare` names.
fn rw_decl(f: &mut FnDecl, t: &HashSet<String>, b: B, owner: &str, method_of: Option<&str>) {
    let key = match method_of { Some(ty) => format!("{}::{}", ty, f.name), None => owner.to_string() };
    let kept: Vec<(usize, Option<Type>)> = f.params.iter().enumerate()
        .filter(|(i, p)| is_target_bare(p, t) && keeps_bare(b, &key, &f.name, *i, method_of.is_some()))
        .map(|(i, p)| (i, p.ty.clone())).collect();
    rw_fn(f, t);
    for (i, ty) in kept { f.params[i].ty = ty; }
}

fn rw_type(ty: &mut Type, t: &HashSet<String>) {
    match ty {
        Type::Named(n) if t.contains(n.as_str()) => {
            *ty = Type::Qualified(Box::new(Type::Named(n.clone())), OwnerQual::Shared);
        }
        Type::Dyn(inner) if matches!(inner.as_ref(), Type::Named(n) if t.contains(n.as_str())) => {
            let n = match inner.as_ref() { Type::Named(n) => n.clone(), _ => unreachable!() };
            *ty = Type::Qualified(Box::new(Type::Named(n)), OwnerQual::Shared);
        }
        Type::Optional(i) | Type::Array(i) | Type::ArrayN(i, _) | Type::ArrayNExpr(i, _)
        | Type::LabeledArray(i, _) | Type::Set(i) | Type::Mut(i) | Type::Dyn(i) => rw_type(i, t),
        Type::Dict(k, v) => { rw_type(k, t); rw_type(v, t); }
        Type::Tuple(es) => for e in es { rw_type(e, t); },
        Type::Generic(_, args) => for a in args { rw_type(a, t); },
        Type::Fn(ret, params, ..) => {
            if let Some(r) = ret { rw_type(r, t); }
            for p in params { rw_type(p, t); }
        }
        // `Trait'owned`, `Trait&`, … were spelled out by the author: leave the qualifier alone.
        Type::Qualified(i, _) if matches!(i.as_ref(), Type::Named(_)) => {}
        Type::Qualified(i, _) => rw_type(i, t),
        _ => {}
    }
}

fn rw_opt(ty: &mut Option<Type>, t: &HashSet<String>) {
    if let Some(ty) = ty { rw_type(ty, t); }
}

fn rw_params(ps: &mut [Param], t: &HashSet<String>) {
    for p in ps { rw_opt(&mut p.ty, t); }
}

fn rw_fn(f: &mut FnDecl, t: &HashSet<String>) {
    rw_params(&mut f.params, t);
    rw_opt(&mut f.return_ty, t);
    rw_opt(&mut f.throws_ty, t);
    rw_stmts(&mut f.body, t);
}

fn rw_stmts(stmts: &mut [Stmt], t: &HashSet<String>) {
    // Both walker callbacks need `t`; neither needs the other's state.
    let mut on_stmt = |s: &mut Stmt| match s {
        Stmt::Let(l) => rw_opt(&mut l.ty, t),
        Stmt::LetDestructure(l) => for b in &mut l.bindings { rw_opt(&mut b.ty, t); },
        Stmt::Fn(f) => {
            rw_params(&mut f.params, t);
            rw_opt(&mut f.return_ty, t);
            rw_opt(&mut f.throws_ty, t);
        }
        _ => {}
    };
    let mut on_expr = |e: &mut Expr| match &mut e.kind {
        ExprKind::Cast(_, ty) => rw_type(ty, t),
        ExprKind::GenericCall(_, tys, _) => for ty in tys { rw_type(ty, t); },
        ExprKind::Closure(params, ret, ..) => {
            rw_params(params, t);
            rw_opt(ret, t);
        }
        _ => {}
    };
    Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(stmts);
}

fn rw_expr(e: &mut Expr, t: &HashSet<String>) {
    let mut noop = |_: &mut Stmt| {};
    let mut on_expr = |e: &mut Expr| match &mut e.kind {
        ExprKind::Cast(_, ty) => rw_type(ty, t),
        ExprKind::GenericCall(_, tys, _) => for ty in tys { rw_type(ty, t); },
        ExprKind::Closure(params, ret, ..) => {
            rw_params(params, t);
            rw_opt(ret, t);
        }
        _ => {}
    };
    Walker { expr: &mut on_expr, stmt: &mut noop }.expr(e);
}

fn rw_items(items: &mut [Item], t: &HashSet<String>, b: B) {
    for item in items {
        match item {
            Item::Fn(f) => { let n = f.name.clone(); rw_decl(f, t, b, &n, None) }
            Item::Struct(s) => {
                for f in &mut s.fields {
                    rw_type(&mut f.ty, t);
                    if let Some(d) = &mut f.default { rw_expr(d, t); }
                }
                for i in &mut s.inits {
                    for p in &mut i.params { rw_opt(&mut p.ty, t); }
                }
                let sn = s.name.clone();
                for m in &mut s.methods { rw_decl(m, t, b, &sn, Some(&sn)); }
                for c in &mut s.conversions { rw_type(&mut c.ty, t); rw_stmts(&mut c.body, t); }
                for st in &mut s.setters { rw_type(&mut st.param_ty, t); rw_stmts(&mut st.body, t); }
                for m in &mut s.type_methods {
                    rw_params(&mut m.params, t);
                    rw_opt(&mut m.return_ty, t);
                    rw_opt(&mut m.throws_ty, t);
                    rw_stmts(&mut m.body, t);
                }
            }
            Item::Enum(e) => {
                for v in &mut e.variants { for f in &mut v.fields { rw_type(&mut f.ty, t); } }
                let en = e.name.clone();
                for m in &mut e.methods { rw_decl(m, t, b, &en, Some(&en)); }
                for c in &mut e.conversions { rw_type(&mut c.ty, t); rw_stmts(&mut c.body, t); }
            }
            Item::Trait(tr) => {
                // Trait contract: a bare parameter is always borrowed (or spelled `'owned`).
                for s in &mut tr.signatures {
                    let kept: Vec<(usize, Option<Type>)> = s.params.iter().enumerate()
                        .filter(|(_, p)| is_target_bare(p, t)).map(|(i, p)| (i, p.ty.clone())).collect();
                    rw_params(&mut s.params, t);
                    for (i, ty) in kept { s.params[i].ty = ty; }
                    rw_opt(&mut s.return_ty, t);
                }
                for d in &mut tr.defaults {
                    let kept: Vec<(usize, Option<Type>)> = d.params.iter().enumerate()
                        .filter(|(_, p)| is_target_bare(p, t)).map(|(i, p)| (i, p.ty.clone())).collect();
                    rw_fn(d, t);
                    for (i, ty) in kept { d.params[i].ty = ty; }
                }
            }
            Item::Ext(x) => { let xn = x.type_name.clone(); for m in &mut x.methods { rw_decl(m, t, b, &xn, Some(&xn)); } }
            Item::Mod(m) => rw_items(&mut m.items, t, b),
            Item::Let(l) => {
                rw_opt(&mut l.ty, t);
                if let Some(v) = &mut l.value { rw_expr(v, t); }
            }
            Item::Alias(a) => rw_type(&mut a.ty, t),
            Item::Stmt(s) => rw_stmts(std::slice::from_mut(s), t),
            Item::Use(_) | Item::Kernel(_) => {}
        }
    }
}
