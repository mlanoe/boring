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
//! methods named like a trait method of the same file or of a file it `use`s (see
//! `set_import_context`), overloaded or generic methods, a parameter also stored somewhere that
//! is not an `'actor` member (a plain or `'shared` member, an element of a `[T'actor]` /
//! `{K = T'actor}` member, `obj.field = p` where `obj`'s struct type is not positively known:
//! a parameter, annotated local or `self` field of a declared struct is), and a parameter whose
//! type differs from the member's own `T'actor` inner type.
use crate::ast::*;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Rewrites `program` in place; see the module doc.
pub fn infer_actor_params(program: &mut Program) {
    let mut trait_methods: HashSet<String> = HashSet::new();
    collect_trait_methods(&program.items, &mut trait_methods);
    // Traits of every file this one reaches through `use` (sibling files, `[deps]`, `boring.*`
    // stdlib): an implementer there must match the trait's `&mut T` signature too.
    IMPORT_CTX.with(|c| {
        if let Some((dir, deps)) = c.borrow().as_ref() {
            let mut visited = HashSet::new();
            collect_imported_trait_methods(&program.items, dir, deps, &mut visited, &mut trait_methods);
        }
    });
    let mut structs: Structs = HashMap::new();
    collect_structs(&program.items, &mut structs);
    rewrite_items(&mut program.items, &trait_methods, &structs);
}

/// Declared fields of every struct of the file, by name (a name declared twice is dropped).
type Structs = HashMap<String, Option<Vec<(String, Type)>>>;

fn collect_structs(items: &[Item], out: &mut Structs) {
    for it in items {
        match it {
            Item::Struct(s) => {
                let fields = s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect();
                out.entry(s.name.clone()).and_modify(|e| *e = None).or_insert(Some(fields));
            }
            Item::Mod(m) => collect_structs(&m.items, out),
            _ => {}
        }
    }
}

/// Inner type name `T` of a member declared `[T'actor]`, `{T'actor}` or `{K = T'actor}`.
fn actor_elem_inner(ty: &Type) -> Option<&str> {
    match ty.without_mut() {
        Type::Array(e) | Type::Set(e) | Type::Dict(_, e) => actor_inner(e),
        _ => None,
    }
}

/// Struct name of a plain `T` / `mut T` type.
fn plain_struct(ty: &Type) -> Option<&str> {
    match ty.without_mut() { Type::Named(n) => Some(n), _ => None }
}

thread_local! {
    /// Directory bare `use` resolves against and the `[deps]` map of the file being built, set by
    /// the entry points before parsing (`set_import_context`). `None`: per-file knowledge only.
    static IMPORT_CTX: std::cell::RefCell<Option<(PathBuf, HashMap<String, PathBuf>)>> =
        const { std::cell::RefCell::new(None) };
}

/// Tells the pass where `use`d files live for the parses that follow on this thread: the
/// directory of `entry` and its project's `[deps]`, resolved like the other import-aware passes.
pub fn set_import_context(entry: &Path) {
    let deps = crate::find_project_root(entry)
        .and_then(|root| {
            let toml_src = std::fs::read_to_string(root.join("boring.toml")).ok()?;
            crate::BoringToml::parse(&toml_src).resolve_deps(&root).ok()
        })
        .unwrap_or_default();
    let dir = entry.parent().map(Path::to_path_buf).unwrap_or_default();
    IMPORT_CTX.with(|c| *c.borrow_mut() = Some((dir, deps)));
}

/// Trait method names of the files `items` `use`, transitively. Imported files are parsed
/// without this pass (`parse_without_inference`): only their declarations matter here.
fn collect_imported_trait_methods(
    items: &[Item],
    source_dir: &Path,
    deps: &HashMap<String, PathBuf>,
    visited: &mut HashSet<PathBuf>,
    out: &mut HashSet<String>,
) {
    for it in items {
        match it {
            Item::Mod(m) => collect_imported_trait_methods(&m.items, source_dir, deps, visited, out),
            Item::Use(u) if !u.path.is_empty() => {
                let (root, source) = if u.path[0] == "boring" && u.path.len() >= 2 {
                    // `use boring.<module>`: embedded stdlib.
                    let Some(src) = crate::stdlib_embed::lookup(&u.path[1]) else { continue };
                    if !visited.insert(PathBuf::from(format!("boring.{}", u.path[1]))) { continue; }
                    (source_dir.to_path_buf(), src.to_string())
                } else {
                    let (root, rel): (&Path, PathBuf) = if u.path.len() >= 2 && deps.contains_key(&u.path[0]) {
                        (&deps[&u.path[0]], u.path[1..].iter().collect())
                    } else {
                        (source_dir, u.path.iter().collect())
                    };
                    let Ok(file) = root.join(rel).with_extension("br").canonicalize() else { continue };
                    if !visited.insert(file.clone()) { continue; }
                    let Ok(src) = std::fs::read_to_string(&file) else { continue };
                    (root.to_path_buf(), src)
                };
                let Ok(tokens) = crate::lexer::lex_all(&source) else { continue };
                let Ok(file) = crate::parser::parse_without_inference(tokens) else { continue };
                collect_trait_methods(&file.items, out);
                collect_imported_trait_methods(&file.items, &root, deps, visited, out);
            }
            _ => {}
        }
    }
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

fn rewrite_items(items: &mut [Item], trait_methods: &HashSet<String>, structs: &Structs) {
    for it in items {
        match it {
            Item::Struct(s) => rewrite_struct(s, trait_methods, structs),
            Item::Mod(m) => rewrite_items(&mut m.items, trait_methods, structs),
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

fn rewrite_struct(s: &mut StructDecl, trait_methods: &HashSet<String>, structs: &Structs) {
    if !s.type_params.is_empty() { return; }
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for m in &s.methods { *counts.entry(m.name.clone()).or_default() += 1; }
    let fields: Vec<(String, Type)> = s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect();
    let struct_name = s.name.clone();
    for m in &mut s.methods {
        if trait_methods.contains(&m.name) || counts[&m.name] != 1 || !m.type_params.is_empty() { continue; }
        for (i, name) in actor_stored_mut_params(m, &fields, structs) {
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
fn actor_stored_mut_params(m: &FnDecl, fields: &[(String, Type)], structs: &Structs) -> Vec<(usize, String)> {
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
    // Struct type of `obj` in `obj.field = p`, where positively known: a parameter or a local
    // with a declared struct type (a name bound twice is unknown), or a `self` field of one.
    let mut local_tys: HashMap<&str, Option<&str>> = HashMap::new();
    for p in &m.params {
        local_tys.insert(&p.name, p.ty.as_ref().and_then(plain_struct));
    }
    walk_stmts(&m.body, &mut |node| {
        if let WalkNode::Stmt(Stmt::Let(l)) = node {
            let ty = l.ty.as_ref().and_then(plain_struct);
            local_tys.entry(&l.name).and_modify(|e| *e = None).or_insert(ty);
        }
    });
    let struct_of = |e: &Expr| -> Option<&str> {
        match &e.kind {
            ExprKind::Var(n) if n != "self" => local_tys.get(n.as_str()).copied().flatten(),
            ExprKind::Field(o, f) if matches!(&o.kind, ExprKind::Var(s) if s == "self") => {
                if m.params.iter().any(|p| &p.name == f) { return None; }
                fields.iter().find(|(n, _)| n == f).and_then(|(_, t)| plain_struct(t))
            }
            _ => None,
        }
    };
    // `obj.field` of a known struct: that struct's declared field type.
    let foreign_member = |e: &Expr| -> Option<&Type> {
        let ExprKind::Field(o, name) = &e.kind else { return None };
        let sname = struct_of(o)?;
        let fs = structs.get(sname)?.as_ref()?;
        fs.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    };
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
                // `d[k] = p` into a `{K = T'actor}` / `[T'actor]` member.
                if let ExprKind::Index(base, _) = &lhs.kind {
                    let elem = member(base).map(|(_, t)| t).or_else(|| foreign_member(base)).and_then(actor_elem_inner);
                    if elem.is_some() && elem == cand_ty(&n) { good.insert(n); } else { bad.insert(n); }
                    return;
                }
                let target = member(lhs).map(|(_, t)| t).or_else(|| foreign_member(lhs));
                match target {
                    Some(fty) if actor_inner(fty) == cand_ty(&n) => { good.insert(n); }
                    _ => { bad.insert(n); }
                }
            }
            ExprKind::MethodCall(recv, meth, args) if STORING.contains(&meth.as_str()) => {
                // An element of a `[T'actor]` / `{K = T'actor}` member holds the caller's handle.
                let elem = member(recv).map(|(_, t)| t).or_else(|| foreign_member(recv)).and_then(actor_elem_inner);
                for a in args {
                    let Some(n) = is_cand(&a.value) else { continue };
                    if elem.is_some() && elem == cand_ty(&n) { good.insert(n); } else { bad.insert(n); }
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
