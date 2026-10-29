//! A `mut` struct parameter of a struct method that the body stores, by bare name, into a
//! member declared `'actor` is inferred `'actor` itself (docs/qualifiers.md, "A `mut`
//! parameter stored in an `'actor` member").
//!
//! A lent `mut T p` is `&mut T`, which cannot outlive the call; storing it can only mean the
//! member and the caller's variable denote the same object, which is what `'actor`
//! (`Arc<Mutex<T>>`) provides. The parameter is rewritten to `mut T'actor p` right after
//! parsing, so the checker, the interpreter and the transpiler all see the same signature: the
//! store check no longer applies (the parameter is no longer lent) and qualifier inference
//! promotes the caller's argument to `'actor` like for any other `'actor` parameter.
//!
//! Left alone (the `check_lent_param_stored` diagnostic keeps applying): trait methods and
//! methods named like a trait method of the same file, overloaded or generic methods, a
//! parameter also stored somewhere that is not an `'actor` member (a plain or `'shared`
//! member, an element of a `[T'actor]`/`{K = T'actor}` member, `obj.field = p` on another
//! object), and a parameter whose type differs from the
//! member's own `T'actor` inner type.
use crate::ast::*;
use std::collections::HashSet;

/// Rewrites `program` in place; see the module doc.
pub fn infer_actor_params(program: &mut Program) {
    let mut trait_methods: HashSet<String> = HashSet::new();
    collect_trait_methods(&program.items, &mut trait_methods);
    rewrite_items(&mut program.items, &trait_methods);
}

fn collect_trait_methods(items: &[Item], out: &mut HashSet<String>) {
    for it in items {
        match it {
            Item::Trait(t) => {
                out.extend(t.signatures.iter().map(|s| s.name.clone()));
                out.extend(t.defaults.iter().map(|d| d.name.clone()));
            }
            Item::Mod(m) => collect_trait_methods(&m.items, out),
            _ => {}
        }
    }
}

fn rewrite_items(items: &mut [Item], trait_methods: &HashSet<String>) {
    for it in items {
        match it {
            Item::Struct(s) => rewrite_struct(s, trait_methods),
            Item::Mod(m) => rewrite_items(&mut m.items, trait_methods),
            _ => {}
        }
    }
}

/// Inner type name `T` of a member declared `T'actor` (a `mut`/`var` wrapper is ignored).
fn actor_inner(ty: &Type) -> Option<&str> {
    match ty.without_mut() {
        Type::Qualified(inner, OwnerQual::Actor) => match inner.without_mut() {
            Type::Named(n) => Some(n),
            _ => None,
        },
        _ => None,
    }
}

fn rewrite_struct(s: &mut StructDecl, trait_methods: &HashSet<String>) {
    if !s.type_params.is_empty() { return; }
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for m in &s.methods { *counts.entry(m.name.clone()).or_default() += 1; }
    let fields: Vec<(String, Type)> = s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect();
    let struct_name = s.name.clone();
    for m in &mut s.methods {
        if trait_methods.contains(&m.name) || counts[&m.name] != 1 || !m.type_params.is_empty() { continue; }
        for (i, name) in actor_stored_mut_params(m, &fields) {
            let p = &mut m.params[i];
            let Some(ty) = p.ty.take() else { continue };
            let (line, col) = (p.line, p.col);
            p.ty = Some(retag(ty));
            crate::transpiler::note_inferred_actor_param(&format!("{}.{}", struct_name, m.name), &name, line, col);
        }
    }
}

fn retag(ty: Type) -> Type {
    match ty {
        Type::Mut(inner) => Type::Mut(Box::new(retag(*inner))),
        other => Type::Qualified(Box::new(other), OwnerQual::Actor),
    }
}

/// `(param index, name)` of each `mut` parameter of `m` stored by bare name into an `'actor`
/// member and nowhere else.
fn actor_stored_mut_params(m: &FnDecl, fields: &[(String, Type)]) -> Vec<(usize, String)> {
    let candidates: Vec<(usize, &Param, &str)> = m.params.iter().enumerate().filter_map(|(i, p)| {
        if !p.mutable || p.rebindable || p.variadic { return None; }
        match p.ty.as_ref()? {
            Type::Mut(inner) => match inner.as_ref() { Type::Named(n) => Some((i, p, n.as_str())), _ => None },
            Type::Named(n) => Some((i, p, n.as_str())),
            _ => None,
        }
    }).collect();
    if candidates.is_empty() { return Vec::new(); }
    // The member a place expression names: `kept`, `self.kept` (a field never shadowed by a parameter).
    let member = |e: &Expr| -> Option<&(String, Type)> {
        let name = match &e.kind {
            ExprKind::Var(n) if !m.params.iter().any(|p| &p.name == n) => n,
            ExprKind::Field(o, n) if matches!(&o.kind, ExprKind::Var(s) if s == "self") => n,
            _ => return None,
        };
        fields.iter().find(|(n, _)| n == name)
    };
    const STORING: &[&str] = &["push", "append", "insert", "add", "set", "pushFront", "push_front", "pushBack", "push_back"];
    // Per candidate: (stored into an actor member, stored anywhere else).
    let mut good: HashSet<String> = HashSet::new();
    let mut bad: HashSet<String> = HashSet::new();
    let cand_ty = |n: &str| candidates.iter().find(|(_, p, _)| p.name == n).map(|(_, _, t)| *t);
    let is_cand = |e: &Expr| -> Option<String> {
        match &e.kind { ExprKind::Var(n) if cand_ty(n).is_some() => Some(n.clone()), _ => None }
    };
    // Any place rooted elsewhere than a candidate (`obj.f = p`, `other[i] = p`) is not an actor store.
    walk_stmts(&m.body, &mut |node| {
        let WalkNode::Expr(e) = node else { return };
        match &e.kind {
            ExprKind::Assign(lhs, rhs) => {
                let Some(n) = is_cand(rhs) else { return };
                match member(lhs) {
                    Some((_, fty)) if actor_inner(fty) == cand_ty(&n) => { good.insert(n); }
                    _ => { bad.insert(n); }
                }
            }
            ExprKind::MethodCall(recv, meth, args) if STORING.contains(&meth.as_str()) => {
                for a in args {
                    let Some(n) = is_cand(&a.value) else { continue };
                    let _ = recv;
                    // Elements of `[T'actor]` / `{K = T'actor}` are not covered: reading them
                    // back is not supported by the transpiler yet, so the diagnostic stays.
                    bad.insert(n);
                }
            }
            _ => {}
        }
    });
    candidates.iter()
        .filter(|(_, p, _)| good.contains(p.name.as_str()) && !bad.contains(p.name.as_str()))
        .map(|(i, p, _)| (*i, p.name.clone()))
        .collect()
}
