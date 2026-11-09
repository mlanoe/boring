//! `mut found = []` whose element type is only implied by later `found.push(x)` calls.
//!
//! `x` being a `Trait'shared` value (a parameter, a `for` variable over a `[Trait'shared]`
//! parameter, or a local with that written type) makes `found` a `[Trait'shared]`; every other
//! push into it (`found.push(Concrete())`) must then be wrapped into the `Arc<dyn Trait>`.
//! Left untyped, rustc infers the Vec type from whichever push it meets first, so the
//! transpiler cannot tell the concrete push needs wrapping. This pass spells the type out
//! (`mut [Trait'shared] found = []`) on a copy of the program the transpiler consumes, so
//! `push`/`insert`/index assignment take their existing typed-collection path.

use std::collections::HashMap;

use crate::ast::*;
use crate::ast_walk::Walker;

fn is_shared(t: &Type) -> bool {
    matches!(t.without_mut(), Type::Qualified(i, OwnerQual::Shared) if matches!(i.as_ref(), Type::Named(_)))
}

fn rw_body(params: &[Param], fields: &[FieldDecl], body: &mut [Stmt]) -> bool {
    let mut env: HashMap<String, Type> = HashMap::new(); // name -> shared element type
    let mut arrays: HashMap<String, Type> = HashMap::new(); // array name -> shared element type
    let param_names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
    for f in fields {
        if param_names.contains(&f.name.as_str()) { continue; }
        if let Type::Array(i) = f.ty.without_mut() {
            if is_shared(i) { arrays.insert(f.name.clone(), i.without_mut().clone()); }
        }
    }
    for p in params {
        match p.ty.as_ref().map(Type::without_mut) {
            Some(t) if is_shared(t) => { env.insert(p.name.clone(), t.clone()); }
            Some(Type::Array(i)) if is_shared(i) => { arrays.insert(p.name.clone(), i.without_mut().clone()); }
            _ => {}
        }
    }
    let mut cands: HashMap<String, Option<Type>> = HashMap::new();
    {
        let mut noop_e = |_: &mut Expr| {};
        let mut on_stmt = |s: &mut Stmt| match s {
            Stmt::Let(l) => match (&l.ty, l.value.as_ref().map(|v| &v.kind)) {
                (None, Some(ExprKind::Array(el))) if el.is_empty() => { cands.insert(l.name.clone(), None); }
                (Some(t), _) if is_shared(t) => { env.insert(l.name.clone(), t.without_mut().clone()); }
                (Some(t), _) => if let Type::Array(i) = t.without_mut() {
                    if is_shared(i) { arrays.insert(l.name.clone(), i.without_mut().clone()); }
                },
                _ => {}
            },
            Stmt::For(f) => {
                let it = match &f.iterable.kind {
                    ExprKind::Var(it) => Some(it),
                    ExprKind::Field(o, it) if matches!(&o.kind, ExprKind::Var(s) if s == "self") => Some(it),
                    _ => None,
                };
                if let (Some(it), [v]) = (it, f.vars.as_slice()) {
                    if let Some(t) = arrays.get(it) { env.insert(v.clone(), t.clone()); }
                }
            }
            _ => {}
        };
        Walker { expr: &mut noop_e, stmt: &mut on_stmt }.stmts(body);
    }
    if cands.is_empty() { return false; }
    {
        let mut noop_s = |_: &mut Stmt| {};
        let mut on_expr = |e: &mut Expr| {
            if let ExprKind::MethodCall(obj, m, args) = &e.kind {
                if m == "push" && args.len() == 1 {
                    if let (ExprKind::Var(arr), ExprKind::Var(x)) = (&obj.kind, &args[0].value.kind) {
                        if let (Some(slot), Some(t)) = (cands.get_mut(arr), env.get(x)) {
                            if slot.is_none() { *slot = Some(t.clone()); }
                        }
                    }
                }
            }
        };
        Walker { expr: &mut on_expr, stmt: &mut noop_s }.stmts(body);
    }
    let mut changed = false;
    let mut noop_e = |_: &mut Expr| {};
    let mut on_stmt = |s: &mut Stmt| {
        if let Stmt::Let(l) = s {
            if l.ty.is_none() {
                if let Some(Some(elem)) = cands.get(&l.name) {
                    if matches!(l.value.as_ref().map(|v| &v.kind), Some(ExprKind::Array(el)) if el.is_empty()) {
                        let arr = Type::Array(Box::new(elem.clone()));
                        l.ty = Some(if l.binding == BindingKind::Mut || l.var_mut { Type::Mut(Box::new(arr)) } else { arr });
                        changed = true;
                    }
                }
            }
        }
    };
    Walker { expr: &mut noop_e, stmt: &mut on_stmt }.stmts(body);
    changed
}

fn rw_items(items: &mut [Item]) -> bool {
    let mut changed = false;
    for item in items {
        match item {
            Item::Fn(f) => changed |= rw_body(&f.params, &[], &mut f.body),
            Item::Struct(s) => { let fl = s.fields.clone(); for m in &mut s.methods { changed |= rw_body(&m.params, &fl, &mut m.body); } }
            Item::Enum(e) => for m in &mut e.methods { changed |= rw_body(&m.params, &[], &mut m.body); },
            Item::Ext(x) => for m in &mut x.methods { changed |= rw_body(&m.params, &[], &mut m.body); },
            Item::Mod(m) => changed |= rw_items(&mut m.items),
            _ => {}
        }
    }
    changed
}

/// A rewritten copy of `program`, or `None` when no local needed its element type spelled out.
pub(crate) fn rewrite_program(program: &Program) -> Option<Program> {
    let mut cloned = program.clone();
    rw_items(&mut cloned.items).then_some(cloned)
}
