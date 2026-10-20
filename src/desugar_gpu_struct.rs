// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of Boring.
// Boring is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// See the LICENSE file at the project root for the full text.

//! Plain-data structs inside device code (kernel bodies and the free functions they call),
//! lowered source-to-source by scalar replacement, so every GPU backend — Metal, wgpu, CUDA,
//! ROCm and simd — inherits struct support without defining struct types in four shader
//! languages (and without layout/padding concerns):
//!
//! * a struct local `p: P` becomes one local per leaf field, `p__a`, `p__n__x` (nested structs
//!   are flattened); `p.a` / `p.n.x` read and write those locals;
//! * `P(a = 1, b = 2.0)` (labeled, positional or defaulted), a copy `q = p`, a field chain and
//!   a call of a struct-returning function all assign leaf by leaf, as does `p = <struct>`;
//! * a free function with a struct parameter and/or return gets device clones: each struct
//!   parameter is spread into its leaves (`f__dev`), and a struct return becomes one clone
//!   per leaf (`f__dev__a`, `f__dev__n__x`). Call sites in device code are rewritten to the
//!   clones; the original stays for host code. Like optional returns, a struct-returning call
//!   is evaluated once per leaf, so it must be free of side effects;
//! * an optional leaf (`int? a`) stays an optional local, which `desugar_gpu_optional` lowers
//!   next on the backends without an `Option`.
//!
//! Eligible structs have only scalar / `bool` / optional-scalar / nested-eligible-struct fields,
//! no type parameters and no custom `init`. Methods are lowered as `S__m(self, ..)` (`req`, or a `def` returning nothing, which returns the updated `self`). Arrays inside structs are not lowered.
//!
//! Struct buffers (`[P]'unified` kernel fields, init parameters, top-level host arrays built
//! from `P(..)` literals) are split into one buffer per leaf field (`ps__a`), so host and device
//! share no struct layout; `ps[i].a` reads `ps__a[i]` on host and device, and `ps[i] = P(..)` /
//! `let q = ps[i]` go leaf by leaf. Not handled: a whole struct buffer passed around as a value
//! (to a function, printed), and buffers whose leaves are optional.

use crate::ast::*;
use crate::ast_walk::Walker;
use std::collections::{HashMap, HashSet};

/// Returns `program` with device-side structs scalar-replaced.
pub fn desugar(program: &Program) -> Program {
    let mut program = program.clone();
    qualify_kernel_helper_calls(&mut program.items);
    let structs = eligible_structs(&program);
    if structs.is_empty() {
        return program;
    }
    let mut cx = Cx { structs, fns: HashMap::new(), env: HashMap::new(), sbufs: HashMap::new() };
    for item in &program.items {
        if let Item::Fn(f) = item {
            if f.qualifier.is_none() && !f.task {
                let params: Vec<Option<String>> = f.params.iter().map(|p| cx.struct_name(p.ty.as_ref())).collect();
                let ret = cx.struct_name(f.return_ty.as_ref());
                if params.iter().any(Option::is_some) || ret.is_some() {
                    cx.fns.insert(f.name.clone(), Sig { params, ret });
                }
            }
        }
    }
    // Struct methods become free functions `S__m(self: S, ..)` (a `def` returning nothing
    // returns the updated `self`), then go through the same clone machinery.
    let mut methods: Vec<FnDecl> = Vec::new();
    let mut method_names: HashMap<String, String> = HashMap::new();
    for (sname, info) in &cx.structs {
        for m in &info.methods {
            if let Some(f) = cx.method_as_fn(sname, m) {
                method_names.insert(f.name.clone(), m.name.clone());
                methods.push(f);
            }
        }
    }
    methods.sort_by(|a, b| a.name.cmp(&b.name));
    for f in &methods {
        let params: Vec<Option<String>> = f.params.iter().map(|p| cx.struct_name(p.ty.as_ref())).collect();
        let ret = cx.struct_name(f.return_ty.as_ref());
        cx.fns.insert(f.name.clone(), Sig { params, ret });
    }
    // Clones are only built for functions a kernel can reach; host-only functions keep their
    // original, struct-typed form (and are not turned into dead, possibly ill-typed twins).
    let (reach_fns, reach_methods) = device_reachable(&program);
    let mut clones = Vec::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            if let Some(sig) = cx.fns.get(&f.name) {
                if reach_fns.contains(&f.name) {
                    clones.extend(cx.device_clones(f, sig));
                }
            }
        }
    }
    for f in &methods {
        if !method_names.get(&f.name).is_some_and(|m| reach_methods.contains(m)) {
            continue;
        }
        let sig = cx.fns[&f.name].clone();
        clones.extend(cx.device_clones(f, &sig));
    }
    // Struct buffers (`[P]` kernel fields, init parameters, host arrays) are split into one
    // buffer per leaf field (structure of arrays), so no struct layout crosses the host/device
    // boundary: `ps[i].a` reads `ps__a[i]` everywhere.
    let mut host_sbufs: HashMap<String, String> = HashMap::new();
    let mut kernel_sbufs: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut field_names: HashMap<String, String> = HashMap::new();
    let mut kernel_names: HashSet<String> = HashSet::new();
    let mut items_out = Vec::with_capacity(program.items.len());
    for item in std::mem::take(&mut program.items) {
        match item {
            Item::Let(l) => {
                let split = l.ty.as_ref().and_then(|t| cx.buffer_struct(t)).and_then(|sn| {
                    let value = l.value.as_ref()?;
                    let leaves = cx.leaves(&sn);
                    let mut lets = Vec::new();
                    for (index, (path, ty)) in leaves.iter().enumerate() {
                        lets.push(LetStmt {
                            name: flat(&l.name, path),
                            ty: l.ty.as_ref().map(|t| retype(t, ty)),
                            value: Some(cx.buffer_leaf_value(value, &sn, index)?),
                            ..l.clone()
                        });
                    }
                    Some((sn, lets))
                });
                match split {
                    Some((sn, lets)) => {
                        host_sbufs.insert(l.name.clone(), sn);
                        items_out.extend(lets.into_iter().map(Item::Let));
                    }
                    None => items_out.push(Item::Let(l)),
                }
            }
            Item::Kernel(k) => {
                kernel_names.insert(k.name.clone());
                let mut map = HashMap::new();
                for f in &k.fields {
                    if let Some(sn) = cx.buffer_struct(&f.ty) {
                        map.insert(f.name.clone(), sn.clone());
                        field_names.insert(f.name.clone(), sn);
                    }
                }
                kernel_sbufs.insert(k.name.clone(), map);
                items_out.push(Item::Kernel(k));
            }
            other => items_out.push(other),
        }
    }
    program.items = items_out;
    for item in &mut program.items {
        if let Item::Kernel(k) = item {
            let sbufs = kernel_sbufs[&k.name].clone();
            for m in &mut k.methods {
                let mut local = Cx { env: HashMap::new(), sbufs: sbufs.clone(), ..cx.clone() };
                m.body = block(std::mem::take(&mut m.body), &mut local);
                rewrite_exprs(&mut m.body, &local);
            }
            if !sbufs.is_empty() {
                cx.split_kernel_decl(k, &sbufs);
            }
        }
    }
    if !host_sbufs.is_empty() || !field_names.is_empty() {
        let host = HostSplit { cx: &cx, host: &host_sbufs, fields: &field_names, kernels: &kernel_names };
        host.rewrite_items(&mut program.items);
    }
    program.items.extend(clones.into_iter().map(Item::Fn));
    program
}

/// Free functions and method names called, transitively, from kernel bodies.
pub(crate) fn device_reachable(program: &Program) -> (HashSet<String>, HashSet<String>) {
    fn calls(body: &[Stmt], fns: &mut HashSet<String>, methods: &mut HashSet<String>) {
        let mut owned = body.to_vec();
        let mut on_expr = |e: &mut Expr| match &e.kind {
            ExprKind::Call(callee, _) => {
                if let ExprKind::Var(name) = &callee.kind {
                    fns.insert(name.clone());
                }
            }
            ExprKind::MethodCall(_, name, _) | ExprKind::Pipe(_, name, _) => {
                methods.insert(name.clone());
            }
            _ => {}
        };
        let mut on_stmt = |_: &mut Stmt| {};
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut owned);
    }
    let mut fns = HashSet::new();
    let mut methods = HashSet::new();
    for item in &program.items {
        if let Item::Kernel(k) = item {
            for m in &k.methods {
                calls(&m.body, &mut fns, &mut methods);
            }
        }
    }
    let by_name: HashMap<&str, &FnDecl> = program
        .items
        .iter()
        .filter_map(|i| match i { Item::Fn(f) => Some((f.name.as_str(), f)), _ => None })
        .collect();
    let struct_methods: Vec<&FnDecl> = program
        .items
        .iter()
        .filter_map(|i| match i { Item::Struct(s) => Some(s.methods.iter()), _ => None })
        .flatten()
        .collect();
    let mut seen_fns: HashSet<String> = HashSet::new();
    let mut seen_methods: HashSet<String> = HashSet::new();
    loop {
        let mut grew = false;
        for name in fns.clone() {
            if seen_fns.insert(name.clone()) {
                grew = true;
                if let Some(f) = by_name.get(name.as_str()) {
                    calls(&f.body, &mut fns, &mut methods);
                }
            }
        }
        for name in methods.clone() {
            if seen_methods.insert(name.clone()) {
                grew = true;
                for m in struct_methods.iter().filter(|m| m.name == name) {
                    calls(&m.body, &mut fns, &mut methods);
                }
            }
        }
        if !grew {
            break;
        }
    }
    (fns, methods)
}

#[derive(Clone)]
struct Sig {
    /// Struct name of each struct-typed parameter.
    params: Vec<Option<String>>,
    /// Struct name of a struct return.
    ret: Option<String>,
}

#[derive(Clone)]
struct StructInfo {
    fields: Vec<(String, Type, Option<Expr>)>,
    methods: Vec<FnDecl>,
}

#[derive(Clone)]
struct Cx {
    structs: HashMap<String, StructInfo>,
    fns: HashMap<String, Sig>,
    /// Struct-typed locals / parameters → struct name.
    env: HashMap<String, String>,
    /// Struct buffers visible here (kernel fields / host arrays) → element struct name.
    sbufs: HashMap<String, String>,
}

/// Where a struct value lives: a local, or an element of a struct buffer.
#[derive(Clone)]
enum Root {
    Var(String),
    Elem(String, Expr),
}

fn unwrap_ty(t: &Type) -> &Type {
    match t {
        Type::Mut(x) | Type::Qualified(x, _) => unwrap_ty(x),
        _ => t,
    }
}

fn is_scalar(t: &Type) -> bool {
    matches!(
        unwrap_ty(t),
        Type::Int | Type::Uint | Type::Uint8 | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64
            | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Float32 | Type::Float64 | Type::Bool
    ) || matches!(unwrap_ty(t), Type::Named(n) if matches!(n.as_str(),
        "int" | "uint" | "float" | "float32" | "float64" | "int8" | "int16" | "int32" | "int64"
        | "uint8" | "uint16" | "uint32" | "uint64" | "bool"))
}

/// Structs whose fields are all scalar, optional-scalar, or another eligible struct.
fn eligible_structs(program: &Program) -> HashMap<String, StructInfo> {
    let decls: HashMap<&str, &StructDecl> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Struct(s) if !s.is_view && !s.is_native && s.type_params.is_empty() && s.inits.is_empty() => {
                Some((s.name.as_str(), s))
            }
            _ => None,
        })
        .collect();
    let mut ok: HashSet<&str> = HashSet::new();
    loop {
        let before = ok.len();
        for (name, s) in &decls {
            if ok.contains(name) {
                continue;
            }
            let fine = s.fields.iter().all(|f| match unwrap_ty(&f.ty) {
                Type::Optional(inner) => is_scalar(inner),
                Type::Named(n) | Type::TypeParam(n) if !is_scalar(&f.ty) => ok.contains(n.as_str()),
                _ => is_scalar(&f.ty),
            });
                    if fine {
                ok.insert(name);
            }
        }
        if ok.len() == before {
            break;
        }
    }
    ok.into_iter()
        .map(|n| {
            let s = decls[n];
            (
                n.to_string(),
                StructInfo {
                    fields: s.fields.iter().map(|f| (f.name.clone(), f.ty.clone(), f.default.clone())).collect(),
                    methods: s.methods.clone(),
                },
            )
        })
        .collect()
}

fn mk(kind: ExprKind, at: &Expr) -> Expr {
    Expr { kind, line: at.line, col: at.col, len: at.len }
}

fn var(name: &str, at: &Expr) -> Expr {
    mk(ExprKind::Var(name.to_string()), at)
}

fn zero(ty: &Type, at: &Expr) -> Expr {
    mk(
        match unwrap_ty(ty) {
            Type::Bool => ExprKind::Bool(false),
            Type::Float32 | Type::Float64 => ExprKind::Float(0.0),
            Type::Named(n) if matches!(n.as_str(), "float" | "float32" | "float64") => ExprKind::Float(0.0),
            _ => ExprKind::Int(0),
        },
        at,
    )
}

fn flat(root: &str, path: &[String]) -> String {
    if path.is_empty() { root.to_string() } else { format!("{root}__{}", path.join("__")) }
}

fn stmt_let(binding: BindingKind, name: &str, ty: Type, value: Expr, line: usize, col: usize) -> Stmt {
    Stmt::Let(LetStmt {
        binding,
        is_pub: false,
        is_static: false,
        name: name.to_string(),
        ty: Some(ty),
        var_mut: false,
        value: Some(value),
        is_lazy: false,
        line,
        col,
    })
}

fn assign(target: Expr, value: Expr) -> Stmt {
    let at = target.clone();
    Stmt::Expr(mk(ExprKind::Assign(Box::new(target), Box::new(value)), &at))
}

impl Cx {
    fn struct_name(&self, t: Option<&Type>) -> Option<String> {
        match t.map(unwrap_ty) {
            // A one-letter struct name (`V`) parses as a type parameter.
            Some(Type::Named(n) | Type::TypeParam(n)) if self.structs.contains_key(n) => Some(n.clone()),
            _ => None,
        }
    }

    /// Leaf fields of struct `s`, as (path, type).
    fn leaves(&self, s: &str) -> Vec<(Vec<String>, Type)> {
        let mut out = Vec::new();
        for (name, ty, _) in &self.structs[s].fields {
            match self.struct_name(Some(ty)) {
                Some(inner) => {
                    for (mut path, t) in self.leaves(&inner) {
                        path.insert(0, name.clone());
                        out.push((path, t));
                    }
                }
                None => out.push((vec![name.clone()], ty.clone())),
            }
        }
        out
    }

    /// `(root, field path)` of a `Var` / `Field` chain, or of one rooted at an element `buf[i]`
    /// of a struct buffer.
    fn path_of(&self, e: &Expr) -> Option<(Root, Vec<String>)> {
        match &e.kind {
            ExprKind::Var(v) => Some((Root::Var(v.clone()), Vec::new())),
            ExprKind::Index(base, idx) => match &base.kind {
                ExprKind::Var(arr) if self.sbufs.contains_key(arr) => Some((Root::Elem(arr.clone(), (**idx).clone()), Vec::new())),
                _ => None,
            },
            ExprKind::Field(obj, f) => {
                let (root, mut path) = self.path_of(obj)?;
                path.push(f.clone());
                Some((root, path))
            }
            _ => None,
        }
    }

    /// The scalar leaf `root.path` as an expression: a local `p__a`, or `buf__a[i]`.
    fn leaf_expr(root: &Root, path: &[String], at: &Expr) -> Expr {
        match root {
            Root::Var(r) => var(&flat(r, path), at),
            Root::Elem(arr, idx) => mk(ExprKind::Index(Box::new(var(&flat(arr, path), at)), Box::new(idx.clone())), at),
        }
    }

    /// `S.m` as the free function `S__m(self: S, ..)`; `None` for methods that cannot be lowered
    /// (task / native / generic, or a `def` that returns a value).
    fn method_as_fn(&self, sname: &str, m: &FnDecl) -> Option<FnDecl> {
        if m.task || m.is_native || m.stream || !m.type_params.is_empty() || m.throws {
            return None;
        }
        let unit_def = m.mutating && m.return_ty.as_ref().is_none_or(|t| matches!(unwrap_ty(t), Type::Void));
        if m.mutating && !unit_def {
            return None;
        }
        let at = Expr { kind: ExprKind::Nil, line: 0, col: 0, len: 0 };
        // Bare field names read `self.field` unless shadowed by a parameter or a local.
        let mut shadowed: HashSet<String> = m.params.iter().map(|p| p.name.clone()).collect();
        let mut names = |stmts: &mut [Stmt]| {
            Walker {
                expr: &mut |_| {},
                stmt: &mut |st| match st {
                    Stmt::Let(l) => {
                        shadowed.insert(l.name.clone());
                    }
                    Stmt::For(f) => shadowed.extend(f.vars.iter().cloned()),
                    _ => {}
                },
            }
            .stmts(stmts);
        };
        let mut body = m.body.clone();
        names(&mut body);
        let fields: HashSet<&str> = self.structs[sname].fields.iter().map(|(n, _, _)| n.as_str()).collect();
        Walker {
            expr: &mut |x| {
                if let ExprKind::Var(n) = &x.kind {
                    if fields.contains(n.as_str()) && !shadowed.contains(n) {
                        *x = mk(ExprKind::Field(Box::new(var("self", x)), n.clone()), x);
                    }
                }
            },
            stmt: &mut |_| {},
        }
        .stmts(&mut body);
        if unit_def {
            body.push(Stmt::Expr(var("self", &at)));
        }
        let mut params = vec![Param {
            name: "self".into(),
            ty: Some(Type::Named(sname.to_string())),
            mutable: false,
            rebindable: false,
            var_mut: false,
            owned: false,
            variadic: false,
            default: None,
            line: 0,
            col: 0,
        }];
        params.extend(m.params.iter().cloned());
        Some(FnDecl {
            name: format!("{sname}__{}", m.name),
            qualifier: None,
            params,
            return_ty: if unit_def { Some(Type::Named(sname.to_string())) } else { m.return_ty.clone() },
            body,
            mutating: false,
            ..m.clone()
        })
    }

    /// `recv.m(args);` where `m` is a `def` returning nothing: (root, path, struct, call).
    fn mutating_method(&self, e: &Expr) -> Option<(Root, Vec<String>, String, Expr)> {
        let ExprKind::MethodCall(recv, _, _) = &e.kind else { return None };
        let call = self.method_call(e)?;
        let sname = self.struct_of(recv)?;
        let ExprKind::Call(callee, _) = &call.kind else { return None };
        let ExprKind::Var(name) = &callee.kind else { return None };
        // Only the unit-`def` form returns its own struct type.
        if self.fns.get(name)?.ret.as_deref() != Some(sname.as_str()) {
            return None;
        }
        let (root, path) = self.path_of(recv)?;
        Some((root, path, sname, call))
    }

    /// `recv.m(args)` on a struct with a lowerable method → `S__m(recv, args)`.
    fn method_call(&self, e: &Expr) -> Option<Expr> {
        let ExprKind::MethodCall(recv, m, args) = &e.kind else { return None };
        let sname = self.struct_of(recv)?;
        let name = format!("{sname}__{m}");
        self.fns.get(&name)?;
        let mut all = vec![Arg { label: None, value: (**recv).clone(), spread: false, default_rest: false }];
        all.extend(args.iter().cloned());
        Some(mk(ExprKind::Call(Box::new(var(&name, e)), all), e))
    }

    /// Struct name of an expression, when it is one.
    fn struct_of(&self, e: &Expr) -> Option<String> {
        if let Some(call) = self.method_call(e) {
            return self.struct_of(&call);
        }
        match &e.kind {
            ExprKind::Call(c, _) => match &c.kind {
                ExprKind::Var(n) if self.structs.contains_key(n) => Some(n.clone()),
                ExprKind::Var(f) => self.fns.get(f).and_then(|s| s.ret.clone()),
                _ => None,
            },
            ExprKind::Var(_) | ExprKind::Field(..) | ExprKind::Index(..) => {
                let (root, path) = self.path_of(e)?;
                let mut s = match &root {
                    Root::Var(v) => self.env.get(v)?.clone(),
                    Root::Elem(arr, _) => self.sbufs.get(arr)?.clone(),
                };
                for field in &path {
                    let ty = self.structs[&s].fields.iter().find(|(n, _, _)| n == field)?.1.clone();
                    s = self.struct_name(Some(&ty))?;
                }
                Some(s)
            }
            _ => None,
        }
    }

    /// One value per leaf of `s` (in `leaves` order) for the struct-valued expression `e`.
    fn field_values(&self, e: &Expr, s: &str) -> Option<Vec<Expr>> {
        if let Some(call) = self.method_call(e) {
            return self.field_values(&call, s);
        }
        match &e.kind {
            ExprKind::Call(c, args) => {
                let ExprKind::Var(name) = &c.kind else { return None };
                if name == s {
                    let info = &self.structs[s];
                    let mut out = Vec::new();
                    for (index, (fname, fty, default)) in info.fields.iter().enumerate() {
                        let given = args
                            .iter()
                            .find(|a| a.label.as_deref() == Some(fname.as_str()))
                            .or_else(|| args.iter().filter(|a| a.label.is_none()).nth(index));
                        let value = given.map(|a| a.value.clone()).or_else(|| default.clone())?;
                        match self.struct_name(Some(fty)) {
                            Some(inner) => out.extend(self.field_values(&value, &inner)?),
                            None => out.push(value),
                        }
                    }
                    return Some(out);
                }
                let sig = self.fns.get(name)?;
                if sig.ret.as_deref() != Some(s) {
                    return None;
                }
                let lowered = self.lower_args(name, args);
                Some(
                    self.leaves(s)
                        .iter()
                        .map(|(path, _)| {
                            mk(
                                ExprKind::Call(Box::new(var(&format!("{name}__dev__{}", path.join("__")), e)), lowered.clone()),
                                e,
                            )
                        })
                        .collect(),
                )
            }
            ExprKind::Var(_) | ExprKind::Field(..) | ExprKind::Index(..) => {
                if self.struct_of(e).as_deref() != Some(s) {
                    return None;
                }
                let (root, path) = self.path_of(e)?;
                Some(self.leaves(s).iter().map(|(p, _)| {
                    let mut full = path.clone();
                    full.extend(p.iter().cloned());
                    Self::leaf_expr(&root, &full, e)
                }).collect())
            }
            _ => None,
        }
    }

    /// Arguments of a call to `f` for its device clone: struct arguments spread into leaves.
    fn lower_args(&self, f: &str, args: &[Arg]) -> Vec<Arg> {
        let Some(sig) = self.fns.get(f) else { return args.to_vec() };
        let mut out = Vec::new();
        for (index, arg) in args.iter().enumerate() {
            let spread = sig.params.get(index).and_then(|p| p.as_ref()).and_then(|s| self.field_values(&arg.value, s));
            match spread {
                Some(values) => out.extend(values.into_iter().map(|v| Arg { label: None, value: v, spread: false, default_rest: false })),
                None => out.push(arg.clone()),
            }
        }
        out
    }

    /// Statements declaring the leaves of `name: s`, initialised from `value`.
    fn declare(&self, binding: &BindingKind, name: &str, s: &str, value: &Expr, line: usize, col: usize) -> Option<Vec<Stmt>> {
        let leaves = self.leaves(s);
        if let ExprKind::If(_) = &value.kind {
            let mut out = Vec::new();
            for (path, ty) in &leaves {
                let init = match unwrap_ty(ty) {
                    Type::Optional(_) => mk(ExprKind::Nil, value),
                    t => zero(t, value),
                };
                out.push(stmt_let(BindingKind::Var, &flat(name, path), ty.clone(), init, line, col));
            }
            out.extend(self.store(&Root::Var(name.to_string()), &[], s, value)?);
            return Some(out);
        }
        let values = self.field_values(value, s)?;
        Some(
            leaves
                .iter()
                .zip(values)
                .map(|((path, ty), v)| stmt_let(binding.clone(), &flat(name, path), ty.clone(), v, line, col))
                .collect(),
        )
    }

    /// Statements assigning the struct-valued `value` into `root.path` (a struct of type `s`).
    fn store(&self, root: &Root, path: &[String], s: &str, value: &Expr) -> Option<Vec<Stmt>> {
        if let ExprKind::If(i) = &value.kind {
            let lower = |body: &Vec<Stmt>| -> Option<Vec<Stmt>> {
                let mut out = body.clone();
                if let Some(Stmt::Expr(last)) = out.pop() {
                    out.extend(self.store(root, path, s, &last)?);
                }
                Some(out)
            };
            let mut branches = Vec::new();
            for (c, b) in &i.branches {
                branches.push((c.clone(), lower(b)?));
            }
            let else_body = match &i.else_body {
                Some(b) => Some(lower(b)?),
                None => None,
            };
            return Some(vec![Stmt::If(IfStmt { branches, else_body, line: i.line, col: i.col })]);
        }
        let values = self.field_values(value, s)?;
        Some(
            self.leaves(s)
                .iter()
                .zip(values)
                .map(|((p, _), v)| {
                    let mut full = path.to_vec();
                    full.extend(p.iter().cloned());
                    assign(Self::leaf_expr(root, &full, value), v)
                })
                .collect(),
        )
    }

    /// The device clone(s) of `f`: struct parameters spread, a struct return split per leaf.
    fn device_clones(&self, f: &FnDecl, sig: &Sig) -> Vec<FnDecl> {
        let mut seeded = self.clone();
        seeded.env.clear();
        let mut params = Vec::new();
        for (p, s) in f.params.iter().zip(&sig.params) {
            match s {
                Some(s) => {
                    seeded.env.insert(p.name.clone(), s.clone());
                    for (path, ty) in self.leaves(s) {
                        params.push(Param {
                            name: flat(&p.name, &path),
                            ty: Some(ty),
                            mutable: false,
                            // Not `var`: a rebindable parameter is a `&mut` out-parameter on the host side.
                            rebindable: false,
                            var_mut: false,
                            default: None,
                            ..p.clone()
                        });
                    }
                }
                None => params.push(p.clone()),
            }
        }
        let build = |name: String, return_ty: Option<Type>, leaf: Option<(&str, usize)>| {
            let mut cx = seeded.clone();
            let mut body = block(f.body.clone(), &mut cx);
            if let Some((s, index)) = leaf {
                map_returns(&mut body, &cx, s, index, true);
            }
            rewrite_exprs(&mut body, &cx);
            // A leaf parameter the body assigns to (a `def` method updating `self`) is copied into
            // a local first: WGSL parameters are immutable and a by-value Rust one needs `mut`.
            let mut assigned: HashSet<String> = HashSet::new();
            Walker {
                expr: &mut |e| {
                    if let ExprKind::Assign(lhs, _) = &e.kind {
                        if let ExprKind::Var(n) = &lhs.kind {
                            assigned.insert(n.clone());
                        }
                    }
                },
                stmt: &mut |_| {},
            }
            .stmts(&mut body);
            let mut params = params.clone();
            let mut prologue = Vec::new();
            for p in &mut params {
                if assigned.contains(&p.name) {
                    let ty = p.ty.clone().unwrap_or(Type::Int);
                    let at = Expr { kind: ExprKind::Nil, line: 0, col: 0, len: 0 };
                    prologue.push(stmt_let(BindingKind::Var, &p.name, ty, var(&format!("{}__in", p.name), &at), 0, 0));
                    p.name = format!("{}__in", p.name);
                }
            }
            prologue.extend(body);
            FnDecl { name, params, return_ty, body: prologue, qualifier: None, ..f.clone() }
        };
        match &sig.ret {
            Some(s) => self
                .leaves(s)
                .into_iter()
                .enumerate()
                .map(|(index, (path, ty))| build(format!("{}__dev__{}", f.name, path.join("__")), Some(ty), Some((s.as_str(), index))))
                .collect(),
            None => vec![build(format!("{}__dev", f.name), f.return_ty.clone(), None)],
        }
    }
}


fn retype(t: &Type, leaf: &Type) -> Type {
    match t {
        Type::Mut(x) => Type::Mut(Box::new(retype(x, leaf))),
        Type::Qualified(x, q) => Type::Qualified(Box::new(retype(x, leaf)), q.clone()),
        Type::Array(_) => Type::Array(Box::new(leaf.clone())),
        Type::ArrayN(_, n) => Type::ArrayN(Box::new(leaf.clone()), *n),
        other => other.clone(),
    }
}

impl Cx {
    /// Element struct of a `[P]` / `[P, N]` buffer type whose leaves are all non-optional.
    fn buffer_struct(&self, t: &Type) -> Option<String> {
        match unwrap_ty(t) {
            Type::Array(e) | Type::ArrayN(e, _) => {
                let s = self.struct_name(Some(e))?;
                self.leaves(&s).iter().all(|(_, ty)| !matches!(unwrap_ty(ty), Type::Optional(_))).then_some(s)
            }
            _ => None,
        }
    }

    /// Leaf `index` of a struct-array initializer: `[P(..), P(..)]` or `[P(..) for ..<n]`.
    fn buffer_leaf_value(&self, value: &Expr, s: &str, index: usize) -> Option<Expr> {
        let leaf = |e: &Expr| self.field_values(e, s).and_then(|mut v| (index < v.len()).then(|| v.swap_remove(index)));
        match &value.kind {
            ExprKind::Array(elems) => {
                Some(mk(ExprKind::Array(elems.iter().map(leaf).collect::<Option<Vec<_>>>()?), value))
            }
            ExprKind::ArrayFill { value: fill, count } => {
                Some(mk(ExprKind::ArrayFill { value: Box::new(leaf(fill)?), count: count.clone() }, value))
            }
            _ => None,
        }
    }

    /// Splits the struct-buffer fields / init parameters of a kernel into leaf buffers.
    fn split_kernel_decl(&self, k: &mut KernelDecl, sbufs: &HashMap<String, String>) {
        let mut fields = Vec::new();
        for f in std::mem::take(&mut k.fields) {
            match sbufs.get(&f.name) {
                Some(sn) => {
                    for (path, ty) in self.leaves(sn) {
                        fields.push(KernelFieldDecl { name: flat(&f.name, &path), ty: retype(&f.ty, &ty), default: None, ..f.clone() });
                    }
                }
                None => fields.push(f),
            }
        }
        k.fields = fields;
        for init in &mut k.inits {
            let mut params = Vec::new();
            let mut split: HashMap<String, String> = HashMap::new();
            for p in std::mem::take(&mut init.params) {
                match p.ty.as_ref().and_then(|t| self.buffer_struct(t)) {
                    Some(sn) => {
                        for (path, ty) in self.leaves(&sn) {
                            params.push(InitParam { name: flat(&p.name, &path), ty: p.ty.as_ref().map(|t| retype(t, &ty)), ..p.clone() });
                        }
                        split.insert(p.name.clone(), sn);
                    }
                    None => params.push(p),
                }
            }
            init.params = params;
            let mut body = Vec::new();
            for st in std::mem::take(&mut init.body) {
                let expanded = match &st {
                    Stmt::Expr(Expr { kind: ExprKind::Assign(lhs, rhs), .. }) => match (&lhs.kind, &rhs.kind) {
                        (ExprKind::Var(f), ExprKind::Var(p)) if sbufs.contains_key(f) && split.contains_key(p) => Some(
                            self.leaves(&sbufs[f])
                                .iter()
                                .map(|(path, _)| assign(var(&flat(f, path), lhs), var(&flat(p, path), rhs)))
                                .collect::<Vec<_>>(),
                        ),
                        _ => None,
                    },
                    _ => None,
                };
                match expanded {
                    Some(stmts) => body.extend(stmts),
                    None => body.push(st),
                }
            }
            init.body = body;
        }
    }
}

/// Host-side rewrites for split struct buffers: `ps[i].a` → `ps__a[i]`, `k.ps[i].a` →
/// `k.ps__a[i]`, and a kernel constructed from `ps` receives its leaf arrays.
struct HostSplit<'a> {
    cx: &'a Cx,
    host: &'a HashMap<String, String>,
    fields: &'a HashMap<String, String>,
    kernels: &'a HashSet<String>,
}

impl HostSplit<'_> {
    fn rewrite_items(&self, items: &mut [Item]) {
        for item in items {
            match item {
                Item::Fn(f) => self.walk(|w| w.stmts(&mut f.body)),
                Item::Stmt(st) => self.walk(|w| w.stmt(st)),
                Item::Let(l) => {
                    if let Some(v) = &mut l.value {
                        self.walk(|w| w.expr(v));
                    }
                }
                Item::Mod(m) => self.rewrite_items(&mut m.items),
                _ => {}
            }
        }
    }

    fn walk(&self, run: impl FnOnce(&mut Walker)) {
        let mut on_expr = |e: &mut Expr| {
            if let Some(new) = self.rewrite(e) {
                *e = new;
            }
        };
        let mut on_stmt = |_: &mut Stmt| {};
        run(&mut Walker { expr: &mut on_expr, stmt: &mut on_stmt });
    }

    fn rewrite(&self, e: &Expr) -> Option<Expr> {
        match &e.kind {
            ExprKind::Field(..) => {
                let mut path = Vec::new();
                let mut cur = e;
                while let ExprKind::Field(obj, f) = &cur.kind {
                    path.insert(0, f.clone());
                    cur = obj;
                }
                let ExprKind::Index(base, idx) = &cur.kind else { return None };
                let (sn, new_base) = match &base.kind {
                    ExprKind::Var(x) => {
                        let sn = self.host.get(x)?;
                        (sn, var(&flat(x, &path), e))
                    }
                    ExprKind::Field(inner, fname) => {
                        let sn = self.fields.get(fname)?;
                        (sn, mk(ExprKind::Field(inner.clone(), flat(fname, &path)), e))
                    }
                    _ => return None,
                };
                self.cx.leaves(sn).iter().any(|(p, _)| *p == path).then(|| mk(ExprKind::Index(Box::new(new_base), idx.clone()), e))
            }
            ExprKind::Call(callee, args) => {
                let ExprKind::Var(k) = &callee.kind else { return None };
                if !self.kernels.contains(k) {
                    return None;
                }
                let mut out = Vec::new();
                let mut changed = false;
                for a in args {
                    match &a.value.kind {
                        ExprKind::Var(x) if self.host.contains_key(x) && a.label.is_none() => {
                            changed = true;
                            for (path, _) in self.cx.leaves(&self.host[x]) {
                                out.push(Arg { label: None, value: var(&flat(x, &path), &a.value), spread: false, default_rest: false });
                            }
                        }
                        _ => out.push(a.clone()),
                    }
                }
                changed.then(|| mk(ExprKind::Call(callee.clone(), out), e))
            }
            _ => None,
        }
    }
}

/// Rewrites `return e` (and the tail value) of a struct-returning function to leaf `index`.
fn map_returns(body: &mut [Stmt], cx: &Cx, s: &str, index: usize, tail: bool) {
    let map = |e: &Expr| -> Expr {
        cx.field_values(e, s).and_then(|mut v| (index < v.len()).then(|| v.swap_remove(index))).unwrap_or_else(|| e.clone())
    };
    let last = body.len().saturating_sub(1);
    for (i, stmt) in body.iter_mut().enumerate() {
        let is_tail = tail && i == last;
        match stmt {
            Stmt::Return(r) => {
                if let Some(v) = &r.value {
                    r.value = Some(map(v));
                }
            }
            Stmt::Expr(e) if is_tail => *e = map(e),
            Stmt::If(i) => {
                for (_, b) in &mut i.branches {
                    map_returns(b, cx, s, index, is_tail);
                }
                if let Some(b) = &mut i.else_body {
                    map_returns(b, cx, s, index, is_tail);
                }
            }
            Stmt::While(w) => map_returns(&mut w.body, cx, s, index, false),
            Stmt::DoWhile(w) => map_returns(&mut w.body, cx, s, index, false),
            Stmt::Loop(l) => map_returns(&mut l.body, cx, s, index, false),
            Stmt::For(f) => map_returns(&mut f.body, cx, s, index, false),
            _ => {}
        }
    }
}

fn block(stmts: Vec<Stmt>, cx: &mut Cx) -> Vec<Stmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for s in stmts {
        match s {
            Stmt::Let(l) => {
                let declared = cx.struct_name(l.ty.as_ref());
                let sname = declared.or_else(|| l.value.as_ref().and_then(|v| cx.struct_of(v)));
                let lowered = match (&sname, &l.value) {
                    (Some(s), Some(v)) => cx.declare(&l.binding, &l.name, s, v, l.line, l.col),
                    _ => None,
                };
                match (sname, lowered) {
                    (Some(s), Some(stmts)) => {
                        cx.env.insert(l.name.clone(), s);
                        out.extend(stmts);
                    }
                    _ => out.push(Stmt::Let(l)),
                }
            }
            Stmt::Expr(e) if matches!(&e.kind, ExprKind::MethodCall(..)) && cx.mutating_method(&e).is_some() => {
                let (root, path, sname, call) = cx.mutating_method(&e).unwrap();
                match cx.store(&root, &path, &sname, &call) {
                    Some(stmts) => out.extend(stmts),
                    None => out.push(Stmt::Expr(e)),
                }
            }
            Stmt::Expr(Expr { kind: ExprKind::Assign(lhs, rhs), line, col, len }) => {
                let lowered = cx.struct_of(&lhs).and_then(|s| {
                    let (root, path) = cx.path_of(&lhs)?;
                    cx.store(&root, &path, &s, &rhs)
                });
                match lowered {
                    Some(stmts) => out.extend(stmts),
                    None => out.push(Stmt::Expr(Expr { kind: ExprKind::Assign(lhs, rhs), line, col, len })),
                }
            }
            Stmt::If(mut i) => {
                for (_, b) in &mut i.branches {
                    *b = block(std::mem::take(b), cx);
                }
                i.else_body = i.else_body.take().map(|b| block(b, cx));
                out.push(Stmt::If(i));
            }
            Stmt::IfLet(mut i) => {
                i.then_body = block(std::mem::take(&mut i.then_body), cx);
                for b in &mut i.elif_branches {
                    b.body = block(std::mem::take(&mut b.body), cx);
                }
                i.else_body = i.else_body.take().map(|b| block(b, cx));
                out.push(Stmt::IfLet(i));
            }
            Stmt::While(mut w) => {
                w.body = block(std::mem::take(&mut w.body), cx);
                out.push(Stmt::While(w));
            }
            Stmt::DoWhile(mut w) => {
                w.body = block(std::mem::take(&mut w.body), cx);
                out.push(Stmt::DoWhile(w));
            }
            Stmt::Loop(mut l) => {
                l.body = block(std::mem::take(&mut l.body), cx);
                out.push(Stmt::Loop(l));
            }
            Stmt::For(mut f) => {
                f.body = block(std::mem::take(&mut f.body), cx);
                out.push(Stmt::For(f));
            }
            other => out.push(other),
        }
    }
    out
}

/// Expression-level rewrites: leaf field reads (`p.a`, `p.n.x` → `p__a`, `p__n__x`) and calls of
/// struct-parameter functions (renamed to their device clone, struct arguments spread).
fn rewrite_exprs(body: &mut [Stmt], cx: &Cx) {
    Walker {
        expr: &mut |e| {
            let replacement: Option<Expr> = match &e.kind {
                ExprKind::Field(..) => match cx.path_of(e) {
                    Some((root, path)) if cx.struct_of(e).is_none() => match &root {
                        Root::Var(v) if cx.env.contains_key(v) => Some(Cx::leaf_expr(&root, &path, e)),
                        Root::Elem(..) => Some(Cx::leaf_expr(&root, &path, e)),
                        _ => None,
                    },
                    _ => None,
                },
                ExprKind::MethodCall(..) => cx.method_call(e).and_then(|call| {
                    let ExprKind::Call(callee, args) = &call.kind else { return None };
                    let ExprKind::Var(f) = &callee.kind else { return None };
                    cx.fns.get(f).filter(|s| s.ret.is_none())?;
                    Some(mk(ExprKind::Call(Box::new(var(&format!("{f}__dev"), callee)), cx.lower_args(f, args)), e))
                }),
                ExprKind::Call(callee, args) => match &callee.kind {
                    ExprKind::Var(f) if cx.fns.get(f).is_some_and(|s| s.ret.is_none()) => Some(mk(
                        ExprKind::Call(Box::new(var(&format!("{f}__dev"), callee)), cx.lower_args(f, args)),
                        e,
                    )),
                    _ => None,
                },
                _ => None,
            };
            if let Some(new) = replacement {
                *e = new;
            }
        },
        stmt: &mut |_| {},
    }
    .stmts(body);
}

/// A device helper declared inside a `kernel` struct (`def float32 twice(float32 x)`) may be
/// called bare (`twice(x)`) from the kernel's `def ()` or from another helper, as the
/// interpreter allows. Every GPU backend only resolves helpers through `self.twice(x)`, so
/// rewrite the bare form to that one before any backend sees it.
fn qualify_kernel_helper_calls(items: &mut [Item]) {
    for item in items {
        match item {
            Item::Kernel(k) => {
                let names: HashSet<String> = k.methods.iter().filter(|m| !m.name.is_empty()).map(|m| m.name.clone()).collect();
                if names.is_empty() {
                    continue;
                }
                for m in &mut k.methods {
                    Walker {
                        expr: &mut |e| {
                            let ExprKind::Call(callee, args) = &e.kind else { return };
                            let ExprKind::Var(f) = &callee.kind else { return };
                            if !names.contains(f) {
                                return;
                            }
                            let recv = var("self", callee);
                            let call = ExprKind::MethodCall(Box::new(recv), f.clone(), args.clone());
                            *e = mk(call, e);
                        },
                        stmt: &mut |_| {},
                    }
                    .stmts(&mut m.body);
                }
            }
            Item::Mod(m) => qualify_kernel_helper_calls(&mut m.items),
            _ => {}
        }
    }
}
