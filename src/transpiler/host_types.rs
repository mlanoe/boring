// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Minimal receiver-type resolution shared by the GPU host emitters (metal/cuda/rocm).
//
// Those emitters rewrite `x.count` / `x.length` / `x.count()` to the array-length builtin
// `.len() as isize`. A receiver whose struct type declares a real field or method of that
// name (`struct Enc: [uint] f; int count` → `enc.count`) must NOT be rewritten. This env
// tracks just enough — struct-typed locals/params, fn and method return types, struct
// field types — to answer "is this receiver a user struct that declares `member`?".

use std::collections::{HashMap, HashSet};

use crate::ast::*;

#[derive(Clone, Default)]
pub(crate) struct HostTypeEnv {
    struct_names: HashSet<String>,
    members: HashMap<String, HashSet<String>>,
    field_structs: HashMap<(String, String), String>,
    fn_returns: HashMap<String, String>,
    method_returns: HashMap<(String, String), String>,
    locals: HashMap<String, String>,
    array_fields: HashSet<(String, String)>,
    array_fns: HashSet<String>,
    array_locals: HashSet<String>,
}

impl HostTypeEnv {
    pub(crate) fn from_program(program: &Program) -> Self {
        let mut env = HostTypeEnv::default();
        for item in &program.items {
            match item {
                Item::Struct(s) => { env.struct_names.insert(s.name.clone()); }
                Item::Enum(e) => { env.struct_names.insert(e.name.clone()); }
                _ => {}
            }
        }
        for item in &program.items {
            match item {
                Item::Struct(s) => {
                    let m = env.members.entry(s.name.clone()).or_default();
                    for f in &s.fields { m.insert(f.name.clone()); }
                    for f in &s.methods { m.insert(f.name.clone()); }
                    for f in &s.fields {
                        if matches!(f.ty.without_mut(), Type::Array(_)) {
                            env.array_fields.insert((s.name.clone(), f.name.clone()));
                        }
                        if let Some(n) = env.type_struct(&f.ty) {
                            env.field_structs.insert((s.name.clone(), f.name.clone()), n);
                        }
                    }
                    for m in &s.methods {
                        if let Some(n) = m.return_ty.as_ref().and_then(|t| env.type_struct(t)) {
                            env.method_returns.insert((s.name.clone(), m.name.clone()), n);
                        }
                    }
                }
                Item::Fn(f) => {
                    if f.return_ty.as_ref().is_some_and(|t| matches!(t.without_mut(), Type::Array(_))) {
                        env.array_fns.insert(f.name.clone());
                    }
                    if let Some(n) = f.return_ty.as_ref().and_then(|t| env.type_struct(t)) {
                        env.fn_returns.insert(f.name.clone(), n);
                    }
                }
                _ => {}
            }
        }
        env
    }

    fn type_struct(&self, ty: &Type) -> Option<String> {
        match ty.without_mut() {
            Type::Named(n) if self.struct_names.contains(n.as_str()) => Some(n.clone()),
            _ => None,
        }
    }

    /// Struct (or enum) type name of `e`, where it can be resolved.
    fn expr_struct(&self, e: &Expr, self_type: Option<&str>) -> Option<String> {
        match &e.kind {
            ExprKind::Var(v) if v == "self" => self_type.map(str::to_string),
            ExprKind::Var(v) => self.locals.get(v.as_str()).cloned(),
            ExprKind::Call(callee, _) => match &callee.kind {
                ExprKind::Var(n) if self.struct_names.contains(n.as_str()) => Some(n.clone()),
                ExprKind::Var(n) => self.fn_returns.get(n.as_str()).cloned(),
                _ => None,
            },
            ExprKind::Field(o, f) => {
                let s = self.expr_struct(o, self_type)?;
                self.field_structs.get(&(s, f.clone())).cloned()
            }
            ExprKind::MethodCall(o, m, _) => {
                let s = self.expr_struct(o, self_type)?;
                self.method_returns.get(&(s, m.clone())).cloned()
            }
            _ => None,
        }
    }

    /// True when `e` is positively known to be a built-in array (`[T]`).
    pub(crate) fn expr_is_array(&self, e: &Expr, self_type: Option<&str>) -> bool {
        match &e.kind {
            ExprKind::Array(_) => true,
            ExprKind::Var(v) => self.array_locals.contains(v.as_str()),
            ExprKind::Call(callee, _) => matches!(&callee.kind, ExprKind::Var(n) if self.array_fns.contains(n.as_str())),
            ExprKind::Field(o, f) => self.expr_struct(o, self_type)
                .is_some_and(|s| self.array_fields.contains(&(s, f.clone()))),
            _ => false,
        }
    }

    /// True when `obj`'s resolved struct type declares a field or method `member`.
    pub(crate) fn is_real_member(&self, obj: &Expr, member: &str, self_type: Option<&str>) -> bool {
        self.expr_struct(obj, self_type)
            .and_then(|s| self.members.get(&s))
            .is_some_and(|m| m.contains(member))
    }

    /// Record a binding's type from its annotation, else from its initializer.
    pub(crate) fn track_local(&mut self, name: &str, ty: Option<&Type>, init: Option<&Expr>, self_type: Option<&str>) {
        let resolved = ty.and_then(|t| self.type_struct(t))
            .or_else(|| init.and_then(|e| self.expr_struct(e, self_type)));
        let is_arr = ty.is_some_and(|t| matches!(t.without_mut(), Type::Array(_)))
            || (ty.is_none() && init.is_some_and(|e| self.expr_is_array(e, self_type)));
        if is_arr { self.array_locals.insert(name.to_string()); } else { self.array_locals.remove(name); }
        match resolved {
            Some(s) => { self.locals.insert(name.to_string(), s); }
            None => { self.locals.remove(name); }
        }
    }

    pub(crate) fn track_param(&mut self, p: &Param) {
        self.track_local(&p.name, p.ty.as_ref(), None, None);
    }

    pub(crate) fn take_locals(&mut self) -> HashMap<String, String> {
        std::mem::take(&mut self.locals)
    }

    pub(crate) fn restore_locals(&mut self, l: HashMap<String, String>) {
        self.locals = l;
    }
}
