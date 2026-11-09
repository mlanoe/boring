// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Type-generic structs and helper functions in device code, as a program-to-program pass shared
// by the Metal, CUDA, ROCm and wgpu targets (it runs first, before `desugar_gpu_struct`, so the
// scalar replacement and every device emitter only ever see concrete, non-generic items).
//
// Shader languages have no Boring-style generics, so each explicit instantiation reachable from
// device code becomes a concrete copy, named like the host monomorphiser names them
// (`monomorphize::mangled_name`): `Pair<float>(a = 1.0, b = 2.0)` and the type `Pair<float>`
// become `Pair_float`, `twice<int>(x)` becomes `twice_int(x)`. The copies are made with the host
// monomorphiser's own substitution (`substitute_struct_decl` / `substitute_fn_decl`) and are
// themselves scanned, so a generic helper may use generic structs and other generic helpers.
//
// Scope: what is scanned is the code a kernel can run: kernel fields, `init`s and methods, the
// free functions they call (transitively) and the methods of structs they call by name. Host-only
// code is left alone (the host emitter has its own specialiser). A generic free function left
// with no remaining use is dropped, so the host does not try to compile a body the device needs.
//
// Not supported, each a Boring error naming the construct (never a shader-compiler error):
// a type-generic kernel (`kernel K<T>`, reported by `kernel_generic`), a generic struct or function
// used in device code without explicit, concrete type arguments (no inference from the arguments:
// `twice(1.5)` must be written `twice<float>(1.5)`), a generic function with a const parameter, and
// a type argument the host mangler cannot name (tuples, dicts, function types).

use std::collections::{HashMap, HashSet};

use crate::ast::*;
use crate::ast_walk::Walker;
use crate::transpiler::monomorphize::{
    build_type_subst, mangled_name, substitute_fn_decl, substitute_struct_decl, substitute_types_in_stmts,
    with_device_generic,
};
use crate::transpiler::TranspileError;

enum Req {
    Struct(String, Vec<Type>),
    Fn(String, Vec<Type>),
}

struct Cx {
    structs: HashMap<String, StructDecl>,
    fns: HashMap<String, FnDecl>,
    names: HashSet<String>,
    made: HashSet<String>,
    queue: Vec<Req>,
    new_items: Vec<Item>,
    errors: Vec<TranspileError>,
    called_fns: HashSet<String>,
    called_methods: HashSet<String>,
}

fn is_ordinary(p: &str) -> bool {
    !p.starts_with('$') && !p.starts_with('^')
}

fn mentions(t: &Type, names: &HashSet<String>) -> Option<String> {
    match t {
        Type::Named(n) if names.contains(n) => Some(n.clone()),
        Type::Generic(n, a) => names.get(n).cloned().or_else(|| a.iter().find_map(|x| mentions(x, names))),
        Type::Tuple(a) => a.iter().find_map(|x| mentions(x, names)),
        Type::Optional(i) | Type::Array(i) | Type::ArrayN(i, _) | Type::Set(i) | Type::Mut(i) | Type::Qualified(i, _) => mentions(i, names),
        Type::Dict(k, v) => mentions(k, names).or_else(|| mentions(v, names)),
        _ => None,
    }
}

pub(crate) fn monomorphise_device_generics(program: &Program) -> (Program, Vec<TranspileError>) {
    // The parser reads a one-letter type in a signature as a type parameter (`add(V a, V b)`), also
    // when a struct `V` is declared: such a name is the struct, not a parameter.
    let declared: HashSet<&str> = program.items.iter().filter_map(|i| match i {
        Item::Struct(s) => Some(s.name.as_str()),
        Item::Enum(e) => Some(e.name.as_str()),
        Item::Alias(a) => Some(a.name.as_str()),
        _ => None,
    }).collect();
    let structs: HashMap<String, StructDecl> = program.items.iter().filter_map(|i| match i {
        Item::Struct(s) if !s.type_params.is_empty() && s.type_params.iter().all(|p| is_ordinary(p) && !declared.contains(p.as_str())) => Some((s.name.clone(), s.clone())),
        _ => None,
    }).collect();
    let fns: HashMap<String, FnDecl> = program.items.iter().filter_map(|i| match i {
        Item::Fn(f) if !f.type_params.is_empty() && f.qualifier.is_none() && f.type_params.iter().all(|p| is_ordinary(p) && !declared.contains(p.as_str())) => Some((f.name.clone(), f.clone())),
        _ => None,
    }).collect();
    if structs.is_empty() && fns.is_empty() {
        return (program.clone(), Vec::new());
    }
    let names: HashSet<String> = structs.keys().cloned().collect();
    let mut cx = Cx {
        structs, fns, names, made: HashSet::new(), queue: Vec::new(), new_items: Vec::new(),
        errors: Vec::new(), called_fns: HashSet::new(), called_methods: HashSet::new(),
    };
    let mut out = program.clone();

    // Roots: every kernel (a type-generic one is `kernel_generic`'s to reject).
    for item in &mut out.items {
        let Item::Kernel(k) = item else { continue };
        if k.type_params.iter().any(|p| is_ordinary(p)) { continue; }
        for f in &mut k.fields {
            f.ty = cx.rename_type(&f.ty);
            if let Some(n) = mentions(&f.ty, &cx.names) {
                cx.errors.push(TranspileError::at(format!("kernel field `{}` has the generic struct `{n}` without concrete type arguments; write `{n}<float>` (device generics are specialised per explicit instantiation)", f.name), f.line, f.col));
            }
        }
        for init in &mut k.inits {
            for p in &mut init.params { if let Some(t) = &p.ty { p.ty = Some(cx.rename_type(t)); } }
            cx.rewrite(&mut init.body);
        }
        for m in &mut k.methods {
            cx.rename_sig(m);
            cx.rewrite(&mut m.body);
        }
    }

    // Worklist: instances to make, non-generic free functions and struct methods to scan.
    let mut scanned_fns: HashSet<usize> = HashSet::new();
    let mut scanned_methods: HashSet<(usize, usize)> = HashSet::new();
    loop {
        let mut grew = false;
        while let Some(req) = cx.queue.pop() {
            grew = true;
            cx.instantiate(req);
        }
        for (idx, item) in out.items.iter_mut().enumerate() {
            match item {
                Item::Fn(f) if f.type_params.is_empty() && f.qualifier.is_none() && cx.called_fns.contains(&f.name) && scanned_fns.insert(idx) => {
                    grew = true;
                    cx.rename_sig(f);
                    cx.rewrite(&mut f.body);
                }
                Item::Struct(s) if s.type_params.is_empty() => {
                    for (mi, m) in s.methods.iter_mut().enumerate() {
                        if cx.called_methods.contains(&m.name) && scanned_methods.insert((idx, mi)) {
                            grew = true;
                            cx.rename_sig(m);
                            cx.rewrite(&mut m.body);
                        }
                    }
                }
                _ => {}
            }
        }
        if !grew && cx.queue.is_empty() { break; }
    }

    out.items.extend(std::mem::take(&mut cx.new_items));
    // A generic declaration nothing refers to any more (every use was device code) is dropped, so
    // the host does not compile a generic body (`a + b` on an unbounded `T`) the device made
    // concrete. What the kept generic declarations refer to is kept too.
    let cands: HashSet<String> = cx.structs.keys().chain(cx.fns.keys()).cloned().collect();
    let is_cand = |i: &Item| match i {
        Item::Struct(s) => cx.structs.contains_key(&s.name) && !s.type_params.is_empty(),
        Item::Fn(f) => cx.fns.contains_key(&f.name) && !f.type_params.is_empty() && f.qualifier.is_none(),
        _ => false,
    };
    let mut keep: HashSet<String> = HashSet::new();
    for item in out.items.iter().filter(|i| !is_cand(i)) { keep.extend(refs_of_item(item, &cands)); }
    loop {
        let before = keep.len();
        for item in out.items.iter().filter(|i| is_cand(i)) {
            let name = match item { Item::Struct(s) => &s.name, Item::Fn(f) => &f.name, _ => continue };
            if keep.contains(name) { keep.extend(refs_of_item(item, &cands)); }
        }
        if keep.len() == before { break; }
    }
    out.items.retain(|i| {
        !is_cand(i) || match i { Item::Struct(s) => keep.contains(&s.name), Item::Fn(f) => keep.contains(&f.name), _ => true }
    });
    (out, cx.errors)
}

impl Cx {
    /// `Name<concrete args>` of a generic struct inside `t` → the instance's name; queues it.
    fn rename_type(&mut self, t: &Type) -> Type {
        let names = self.names.clone();
        let (t, uses) = with_device_generic(names, || crate::transpiler::monomorphize::substitute_type(t, &HashMap::new()));
        for (n, a) in uses { self.queue.push(Req::Struct(n, a)); }
        t
    }

    fn rename_sig(&mut self, f: &mut FnDecl) {
        for p in &mut f.params { if let Some(t) = &p.ty { p.ty = Some(self.rename_type(t)); } }
        if let Some(t) = &f.return_ty { f.return_ty = Some(self.rename_type(t)); }
        for p in f.params.iter().chain(std::iter::empty()) {
            if let Some(n) = p.ty.as_ref().and_then(|t| mentions(t, &self.names)) {
                self.errors.push(TranspileError::at(format!("parameter `{}` of `{}` has the generic struct `{n}` without concrete type arguments; write `{n}<float>` (device generics are specialised per explicit instantiation)", p.name, f.name), p.line, p.col));
            }
        }
    }

    /// Renames generic types in `body`, turns explicit generic constructions and calls into plain
    /// ones of the instance, and reports every use that has no concrete type arguments.
    fn rewrite(&mut self, body: &mut [Stmt]) {
        let names = self.names.clone();
        let (_, uses) = with_device_generic(names, || substitute_types_in_stmts(body, &HashMap::new()));
        for (n, a) in uses { self.queue.push(Req::Struct(n, a)); }

        let mut found: Vec<Req> = Vec::new();
        let mut errors: Vec<TranspileError> = Vec::new();
        let mut stmt_errors: Vec<TranspileError> = Vec::new();
        let mut called_fns: HashSet<String> = HashSet::new();
        let mut called_methods: HashSet<String> = HashSet::new();
        {
            let structs = &self.structs;
            let fns = &self.fns;
            let names = &self.names;
            let mut on_expr = |e: &mut Expr| {
                match &mut e.kind {
                    ExprKind::GenericCall(callee, targs, args) => {
                        let ExprKind::Var(name) = &callee.kind else { return };
                        let params: Option<&Vec<String>> = structs.get(name).map(|s| &s.type_params).or_else(|| fns.get(name).map(|f| &f.type_params));
                        let Some(params) = params else { return };
                        let want = params.iter().filter(|p| is_ordinary(p)).count();
                        let Some(mangled) = (targs.len() == want).then(|| mangled_name(name, targs)).flatten()
                            .filter(|_| !targs.iter().any(|t| matches!(t, Type::TypeParam(_)))) else {
                            errors.push(TranspileError::at(format!("`{name}` takes {want} type argument(s); in device code they must be given explicitly and be concrete scalar, struct or array types (`{name}<float>`)"), e.line, e.col));
                            return;
                        };
                        found.push(if structs.contains_key(name) { Req::Struct(name.clone(), targs.clone()) } else { Req::Fn(name.clone(), targs.clone()) });
                        let taken = std::mem::take(args);
                        let (line, col, len) = (callee.line, callee.col, callee.len);
                        *e = Expr {
                            kind: ExprKind::Call(Box::new(Expr { kind: ExprKind::Var(mangled), line, col, len }), taken),
                            line: e.line, col: e.col, len: e.len,
                        };
                    }
                    ExprKind::Call(callee, _) => {
                        let ExprKind::Var(name) = &callee.kind else { return };
                        if let Some(s) = structs.get(name) {
                            // An untyped-field struct (`T_<field>`) is D6's: its own message follows.
                            if !s.type_params.iter().any(|p| p.starts_with("T_")) {
                                errors.push(TranspileError::at(format!("generic struct `{name}` is built in device code without type arguments; write `{name}<float>(..)` (device generics are specialised per explicit instantiation, never inferred)"), e.line, e.col));
                            }
                        } else if fns.contains_key(name) {
                            errors.push(TranspileError::at(format!("generic function `{name}` is called from device code without type arguments; write `{name}<float>(..)` (device generics are specialised per explicit instantiation, never inferred)"), e.line, e.col));
                        } else {
                            called_fns.insert(name.clone());
                        }
                    }
                    ExprKind::MethodCall(_, name, _) | ExprKind::Pipe(_, name, _) => { called_methods.insert(name.clone()); }
                    _ => {}
                }
            };
            let mut on_stmt = |s: &mut Stmt| {
                if let Stmt::Let(l) = s {
                    if let Some(n) = l.ty.as_ref().and_then(|t| mentions(t, names)) {
                        stmt_errors.push(TranspileError::at(format!("local `{}` is typed with the generic struct `{n}` without concrete type arguments; write `{n}<float>` (device generics are specialised per explicit instantiation)", l.name), l.line, l.col));
                    }
                }
            };
            Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(body);
        }
        self.queue.extend(found);
        self.errors.extend(errors);
        self.errors.extend(stmt_errors);
        self.called_fns.extend(called_fns);
        self.called_methods.extend(called_methods);
    }

    fn instantiate(&mut self, req: Req) {
        match req {
            Req::Struct(name, args) => {
                let Some(mangled) = mangled_name(&name, &args) else {
                    self.errors.push(TranspileError::at(format!("generic struct `{name}`: a type argument cannot be named for a device specialisation (use scalar, struct or array types)"), 0, 0));
                    return;
                };
                if !self.made.insert(mangled.clone()) { return; }
                let decl = self.structs[&name].clone();
                if decl.type_params.len() != args.len() { return; }
                let subst = build_type_subst(&decl.type_params, &args);
                let names = self.names.clone();
                let (mut spec, uses) = with_device_generic(names, || substitute_struct_decl(&decl, &subst, &mangled));
                for (n, a) in uses { self.queue.push(Req::Struct(n, a)); }
                for m in &mut spec.methods { self.rewrite(&mut m.body); }
                for i in &mut spec.inits { self.rewrite(&mut i.body); }
                self.new_items.push(Item::Struct(spec));
            }
            Req::Fn(name, args) => {
                let Some(mangled) = mangled_name(&name, &args) else { return };
                if !self.made.insert(mangled.clone()) { return; }
                let decl = self.fns[&name].clone();
                let subst = build_type_subst(&decl.type_params, &args);
                let mut spec = substitute_fn_decl(&decl, &subst, &mangled);
                self.rename_sig(&mut spec);
                self.rewrite(&mut spec.body);
                self.new_items.push(Item::Fn(spec));
            }
        }
    }
}

fn type_names(t: &Type, out: &mut HashSet<String>) {
    match t {
        Type::Named(n) => { out.insert(n.clone()); }
        Type::Generic(n, a) => { out.insert(n.clone()); for x in a { type_names(x, out); } }
        Type::Tuple(a) => for x in a { type_names(x, out); },
        Type::Optional(i) | Type::Array(i) | Type::ArrayN(i, _) | Type::Set(i) | Type::Mut(i) | Type::Qualified(i, _) => type_names(i, out),
        Type::Dict(k, v) => { type_names(k, out); type_names(v, out); }
        _ => {}
    }
}

/// The names of `cands` that `item` mentions, as a value (`Name(..)`, `name<T>(..)`) or in a type.
fn refs_of_item(item: &Item, cands: &HashSet<String>) -> HashSet<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut item = item.clone();
    {
        let mut on_expr = |e: &mut Expr| {
            if let ExprKind::Var(n) = &e.kind { seen.insert(n.clone()); }
        };
        let mut types: Vec<Type> = Vec::new();
        let mut on_stmt = |s: &mut Stmt| {
            if let Stmt::Let(l) = s { if let Some(t) = &l.ty { types.push(t.clone()); } }
        };
        let mut sig_types: Vec<Type> = Vec::new();
        let fn_sig = |f: &FnDecl, out: &mut Vec<Type>| {
            out.extend(f.params.iter().filter_map(|p| p.ty.clone()));
            out.extend(f.return_ty.clone());
        };
        let mut w = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
        match &mut item {
            Item::Fn(f) => { fn_sig(f, &mut sig_types); w.stmts(&mut f.body); }
            Item::Struct(s) => {
                sig_types.extend(s.fields.iter().map(|f| f.ty.clone()));
                for m in &mut s.methods { fn_sig(m, &mut sig_types); w.stmts(&mut m.body); }
                for i in &mut s.inits { w.stmts(&mut i.body); }
            }
            Item::Kernel(k) => {
                sig_types.extend(k.fields.iter().map(|f| f.ty.clone()));
                for i in &mut k.inits { sig_types.extend(i.params.iter().filter_map(|p| p.ty.clone())); w.stmts(&mut i.body); }
                for m in &mut k.methods { fn_sig(m, &mut sig_types); w.stmts(&mut m.body); }
            }
            Item::Let(l) => { sig_types.extend(l.ty.clone()); if let Some(v) = &mut l.value { w.expr(v); } }
            other => crate::transpiler::kernel_generic::walk_item(other, &mut w),
        }
        for t in types.iter().chain(&sig_types) { type_names(t, &mut seen); }
    }
    seen.retain(|n| cands.contains(n));
    seen
}
