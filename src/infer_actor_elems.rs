//! `[mut T]` / `{K = mut V}` elements of a struct type are inferred `'actor` when the file takes
//! an alias of an element (docs/qualifiers.md, "Content-mutable collection elements").
//!
//! `let q = arr[0]` followed by `arr[0].x = 9.0` has one natural reading, the interpreter's: `q`
//! and the element are the same object. A plain `Vec<T>` cannot express that (an index read is a
//! copy, and a `&arr[0]` borrow would forbid the later write), a shared handle can. So whenever a
//! file binds an element read by `let`/`mut`/`var` (`let q = coll[i]`), every declared
//! `[mut T]` / `{K = mut V}` type in it whose element is a struct declared in the same file is
//! rewritten to `[mut T'actor]` / `{K = mut V'actor}` right after parsing, like
//! `infer_actor_params`: the checker, the interpreter and the transpiler all see the same type.
//!
//! Off for `boring build --linux` (see `disable`).
//!
//! Left alone: an element type with an explicit qualifier (`[mut T'inline]` opts out), a generic
//! struct, a struct not declared in this file, a struct a kernel mentions (or a field type of one: GPU
//! data stays plain), and a collection type with its own outer qualifier (`[mut T]'unified`). Without `mut` on the element nothing can be written through the alias, so
//! plain `[T]` is never rewritten.
use crate::ast::*;
use crate::ast_walk::Walker;
use std::collections::HashSet;

thread_local! {
    static DISABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Turns the pass off for the parses that follow on this thread: the `--linux` (`no_std`) emitter
/// has no lock-aware element access, so promoted elements would not compile there.
pub fn disable() { DISABLED.with(|d| d.set(true)); }

pub fn infer_actor_elems(program: &mut Program) {
    if DISABLED.with(|d| d.get()) { return; }
    let mut structs: HashSet<String> = HashSet::new();
    let mut kernel_text = String::new();
    let mut decls: Vec<(String, String)> = Vec::new();
    collect_decls(&program.items, &mut structs, &mut kernel_text, &mut decls);
    // A struct a kernel mentions, or a field type of one (transitively), is device data: it stays plain.
    let mut changed = true;
    while changed {
        changed = false;
        for (name, text) in &decls {
            if structs.contains(name) && kernel_text.contains(&format!("Named({name:?})")) {
                kernel_text.push_str(text);
                structs.remove(name);
                changed = true;
            }
        }
    }
    if structs.is_empty() { return; }
    let mut aliased = false;
    visit_bodies(&mut program.items, &mut |stmt| {
        if let Stmt::Let(l) = stmt {
            if matches!(&l.value, Some(Expr { kind: ExprKind::Index(..), .. })) { aliased = true; }
        }
    });
    if !aliased { return; }
    rewrite_items(&mut program.items, &structs);
}

fn collect_decls(items: &[Item], structs: &mut HashSet<String>, kernel_text: &mut String, decls: &mut Vec<(String, String)>) {
    let mut seen: HashSet<String> = HashSet::new();
    for it in items {
        match it {
            Item::Struct(s) => {
                // A name declared twice is ambiguous: dropped.
                if !seen.insert(s.name.clone()) { structs.remove(&s.name); }
                else if s.type_params.is_empty() && !s.is_view && !s.is_native { structs.insert(s.name.clone()); }
                decls.push((s.name.clone(), format!("{:?}", s.fields)));
            }
            Item::Kernel(k) => kernel_text.push_str(&format!("{k:?}")),
            Item::Mod(m) => collect_decls(&m.items, structs, kernel_text, decls),
            _ => {}
        }
    }
}

fn visit_bodies(items: &mut [Item], f: &mut dyn FnMut(&mut Stmt)) {
    for it in items {
        match it {
            Item::Fn(d) => walk(&mut d.body, f),
            Item::Struct(s) => {
                for m in &mut s.methods { walk(&mut m.body, f); }
                for i in &mut s.inits { walk(&mut i.body, f); }
            }
            Item::Ext(e) => { for m in &mut e.methods { walk(&mut m.body, f); } }
            Item::Mod(m) => visit_bodies(&mut m.items, f),
            Item::Stmt(s) => walk(std::slice::from_mut(s), f),
            _ => {}
        }
    }
}

fn walk(stmts: &mut [Stmt], f: &mut dyn FnMut(&mut Stmt)) {
    Walker { expr: &mut |_| {}, stmt: f }.stmts(stmts);
}

fn rewrite_items(items: &mut [Item], structs: &HashSet<String>) {
    for it in items.iter_mut() {
        match it {
            Item::Fn(d) => rewrite_fn(d, structs),
            Item::Struct(s) => {
                for f in &mut s.fields { rewrite_type(&mut f.ty, structs); }
                for m in &mut s.methods { rewrite_fn(m, structs); }
                for i in &mut s.inits {
                    for p in &mut i.params { if let Some(t) = &mut p.ty { rewrite_type(t, structs); } }
                }
            }
            Item::Ext(e) => { for m in &mut e.methods { rewrite_fn(m, structs); } }
            Item::Let(l) => { if let Some(t) = &mut l.ty { rewrite_type(t, structs); } }
            Item::Mod(m) => rewrite_items(&mut m.items, structs),
            _ => {}
        }
    }
    visit_bodies(items, &mut |stmt| {
        if let Stmt::Let(l) = stmt {
            if let Some(t) = &mut l.ty { rewrite_type(t, structs); }
        }
    });
}

fn rewrite_fn(d: &mut FnDecl, structs: &HashSet<String>) {
    if d.type_params.is_empty() {
        for p in &mut d.params { if let Some(t) = &mut p.ty { rewrite_type(t, structs); } }
        if let Some(t) = &mut d.return_ty { rewrite_type(t, structs); }
    }
}

/// `mut T` of a declared struct `T` -> `mut T'actor`.
fn actor_elem(e: &mut Type, structs: &HashSet<String>) {
    if let Type::Mut(inner) = e {
        if let Type::Named(n) = inner.as_ref() {
            if structs.contains(n) {
                let named = std::mem::replace(inner.as_mut(), Type::Void);
                **inner = Type::Qualified(Box::new(named), OwnerQual::Actor);
            }
        }
    } else {
        rewrite_type(e, structs);
    }
}

fn rewrite_type(ty: &mut Type, structs: &HashSet<String>) {
    match ty {
        Type::Array(e) => actor_elem(e, structs),
        Type::Dict(_, v) => actor_elem(v, structs),
        Type::Optional(i) | Type::Mut(i) => rewrite_type(i, structs),
        Type::Tuple(ts) => { for t in ts { rewrite_type(t, structs); } }
        Type::Set(e) => rewrite_type(e, structs),
        _ => {}
    }
}
