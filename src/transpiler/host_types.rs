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
    ctors: HashMap<String, StructCtor>,
}

/// How a call `Name(args)` to a user struct is constructed in host code: a literal
/// `Name { field: value, .. }` (declared fields, in order), or `Name::new(args)` when the
/// struct has an `init` with a body (computed fields can't be written in a literal).
#[derive(Clone)]
struct StructCtor {
    fields: Vec<(String, Type, Option<Expr>)>,
    via_new: bool,
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
                    env.register_ctor(s);
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

    fn register_ctor(&mut self, s: &StructDecl) {
        if !s.type_params.is_empty() || s.is_view || s.is_native { return; }
        let mut fields: Vec<(String, Type, Option<Expr>)> =
            s.fields.iter().map(|f| (f.name.clone(), f.ty.clone(), f.default.clone())).collect();
        if fields.is_empty() {
            for init in s.inits.iter().filter(|i| i.body.is_empty()) {
                for p in &init.params {
                    if let Some(ty) = &p.ty { fields.push((p.name.clone(), ty.clone(), None)); }
                }
            }
        }
        let via_new = s.inits.iter().any(|i| !i.body.is_empty());
        self.ctors.insert(s.name.clone(), StructCtor { fields, via_new });
    }

    /// Rust text for the construction call `name(args)` when `name` is a user struct, so a
    /// host emitter does not fall through to a plain function call `Name(..)` (E0423).
    /// `value` renders one argument for a field of the given declared type. `None` when
    /// `name` is not a plain constructible struct or the call shape isn't understood
    /// (spread, `_` fill, unknown label, missing field without a default).
    pub(crate) fn struct_ctor_call(
        &self,
        name: &str,
        args: &[Arg],
        value: &mut dyn FnMut(&Type, &Expr) -> String,
    ) -> Option<String> {
        let ctor = self.ctors.get(name)?;
        if args.iter().any(|a| a.spread || a.default_rest) { return None; }
        if ctor.via_new {
            let rendered: Vec<String> = args.iter().enumerate().map(|(i, a)| match ctor.fields.get(i) {
                Some((_, ty, _)) => value(ty, &a.value),
                None => value(&Type::Int, &a.value),
            }).collect();
            return Some(format!("{}::new({})", name, rendered.join(", ")));
        }
        if args.len() > ctor.fields.len() { return None; }
        let mut given: Vec<Option<&Expr>> = vec![None; ctor.fields.len()];
        for (i, a) in args.iter().enumerate() {
            let idx = match &a.label {
                Some(l) => ctor.fields.iter().position(|(n, _, _)| n == l)?,
                None => i,
            };
            if given[idx].is_some() { return None; }
            given[idx] = Some(&a.value);
        }
        let mut parts = Vec::new();
        for ((fname, fty, default), g) in ctor.fields.iter().zip(given) {
            let e = g.or(default.as_ref())?;
            parts.push(format!("{}: {}", fname, value(fty, e)));
        }
        Some(format!("{} {{ {} }}", name, parts.join(", ")))
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

/// Free-function math builtins (`abs(x)`, `sqrt(x)`, …) as a Rust method call on the already
/// emitted argument. The GPU host emitters are separate codegen paths from the general
/// `emit_builtin_call` and don't delegate to it, so without this a call fell through to a bare
/// `abs(...)` (rustc E0425). `float_literal` arguments get an `f64` cast: an unsuffixed literal
/// receiver is ambiguous (E0689). Returns `None` for any other callee.
pub(crate) fn gpu_host_math_builtin(name: &str, arg: &Expr, inner: &str) -> Option<String> {
    let method = match name {
        "sqrt" | "abs" | "floor" | "ceil" | "round" | "sin" | "cos" | "tan"
        | "asin" | "acos" | "atan" | "exp" | "tanh" | "log2" | "log10" => name,
        "log" => "ln",
        _ => return None,
    };
    let recv = if matches!(arg.kind, ExprKind::Float(_)) { format!("({inner} as f64)") } else { format!("({inner})") };
    Some(format!("{recv}.{method}()"))
}
