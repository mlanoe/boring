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
    visit_fn_bodies(&mut program.items, &mut |body| {
        if body_aliases_element(body) { aliased = true; }
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

/// Calls `f` with every function/method/init body, and with each top-level statement as a body of its own.
fn visit_fn_bodies(items: &mut [Item], f: &mut dyn FnMut(&mut [Stmt])) {
    for it in items {
        match it {
            Item::Fn(d) => f(&mut d.body),
            Item::Struct(s) => {
                for m in &mut s.methods { f(&mut m.body); }
                for i in &mut s.inits { f(&mut i.body); }
            }
            Item::Ext(e) => { for m in &mut e.methods { f(&mut m.body); } }
            Item::Mod(m) => visit_fn_bodies(&mut m.items, f),
            Item::Stmt(s) => f(std::slice::from_mut(s)),
            _ => {}
        }
    }
}

fn is_loop(stmt: &Stmt) -> bool {
    matches!(stmt, Stmt::While(_) | Stmt::WhileLet(_) | Stmt::DoWhile(_) | Stmt::Loop(_) | Stmt::For(_))
}

/// Whether `body` binds an element read (`let q = arr[i]`) in a way another name can observe. A
/// bind whose source is a plain local declared in the same body, outside any loop, and never
/// mentioned again after the bind (the element is only edited through the new name and pushed
/// elsewhere, as in `mut t = inner[0]; t.name = "x"; out.push(t)`) has no second name to alias.
fn body_aliases_element(body: &mut [Stmt]) -> bool {
    use std::cell::RefCell;
    // Binds inside any loop body: the loop's next iteration is a "later" use.
    let in_loop: RefCell<HashSet<*const Stmt>> = RefCell::new(HashSet::new());
    walk(body, &mut |stmt| {
        if is_loop(stmt) {
            let mut inner: Vec<*const Stmt> = Vec::new();
            let mut collect = |s: &mut Stmt| { if matches!(s, Stmt::Let(_)) { inner.push(s as *const Stmt); } };
            match stmt {
                Stmt::While(w) => walk(&mut w.body, &mut collect),
                Stmt::WhileLet(w) => walk(&mut w.body, &mut collect),
                Stmt::DoWhile(d) => walk(&mut d.body, &mut collect),
                Stmt::Loop(l) => walk(&mut l.body, &mut collect),
                Stmt::For(f) => walk(&mut f.body, &mut collect),
                _ => {}
            }
            in_loop.borrow_mut().extend(inner);
        }
    });
    struct State {
        declared: HashSet<String>,
        /// Local roots bound from so far, each still waiting for a later mention.
        pending: HashSet<String>,
        skip: Option<String>,
        aliased: bool,
    }
    let st = RefCell::new(State { declared: HashSet::new(), pending: HashSet::new(), skip: None, aliased: false });
    Walker {
        expr: &mut |e| {
            if let ExprKind::Var(n) = &e.kind {
                let mut st = st.borrow_mut();
                if st.skip.as_deref() == Some(n.as_str()) { st.skip = None; }
                else if st.pending.contains(n) { st.aliased = true; }
            }
        },
        stmt: &mut |stmt| {
            let ptr = stmt as *const Stmt;
            let Stmt::Let(l) = stmt else { return };
            let mut st = st.borrow_mut();
            if let Some(Expr { kind: ExprKind::Index(obj, _), .. }) = &l.value {
                match &obj.kind {
                    ExprKind::Var(root) if st.declared.contains(root) && !in_loop.borrow().contains(&ptr) => {
                        st.pending.insert(root.clone());
                        st.skip = Some(root.clone());
                    }
                    _ => st.aliased = true,
                }
            }
            // A later `let` of the same name (a sibling branch's own scratch array) is a new variable.
            st.pending.remove(&l.name);
            st.declared.insert(l.name.clone());
        },
    }.stmts(body);
    st.into_inner().aliased
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
