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

use super::*;
use super::Transpiler;
use super::helpers::*;
use crate::ast_walk::Walker;

#[derive(Default)]
struct AmbientUsage {
    has_time: bool,
    has_input: bool,
    has_commands: bool,
    has_entity: bool,
    resources: Vec<String>,
    mut_resources: Vec<String>,
}

fn scan_ambient_usage(body: &[Stmt], known_resources: &std::collections::HashSet<String>) -> AmbientUsage {
    let mut usage = AmbientUsage::default();
    let mut on_stmt = |_s: &mut Stmt| {};
    let mut on_expr = |e: &mut Expr| {
        match &e.kind {
            ExprKind::DotIdent(name) => {
                match name.as_str() {
                    "time" => usage.has_time = true,
                    "input" => usage.has_input = true,
                    "entity" => usage.has_entity = true,
                    _ => {
                        if known_resources.contains(name) {
                            if !usage.resources.contains(name) && !usage.mut_resources.contains(name) {
                                usage.resources.push(name.clone());
                            }
                        }
                    }
                }
            }
            ExprKind::Field(base, _field) => {
                if let ExprKind::DotIdent(name) = &base.kind {
                    match name.as_str() {
                        "time" => usage.has_time = true,
                        "input" => usage.has_input = true,
                        _ => {}
                    }
                }
            }
            ExprKind::Var(name) => {
                if known_resources.contains(name) {
                    if !usage.resources.contains(name) && !usage.mut_resources.contains(name) {
                        usage.resources.push(name.clone());
                    }
                }
            }
            ExprKind::Assign(target, _) => {
                let var_name = match &target.kind {
                    ExprKind::Var(n) => Some(n.clone()),
                    ExprKind::DotIdent(n) => Some(n.clone()),
                    ExprKind::Field(base, _) => match &base.kind {
                        ExprKind::Var(n) => Some(n.clone()),
                        ExprKind::DotIdent(n) => Some(n.clone()),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(vn) = var_name {
                    if known_resources.contains(&vn) {
                        if let Some(pos) = usage.resources.iter().position(|r| r == &vn) {
                            usage.resources.remove(pos);
                        }
                        if !usage.mut_resources.contains(&vn) {
                            usage.mut_resources.push(vn);
                        }
                    }
                }
            }
            ExprKind::Call(callee, args) => {
                if let ExprKind::DotIdent(name) = &callee.kind {
                    match name.as_str() {
                        "despawn" => {
                            usage.has_commands = true;
                            usage.has_entity = true;
                        }
                        "spawn" => {
                            usage.has_commands = true;
                        }
                        "resource" => {
                            if let Some(first) = args.first() {
                                if let ExprKind::Var(res_name) = &first.value.kind {
                                    if !usage.resources.contains(res_name) && !usage.mut_resources.contains(res_name) {
                                        usage.resources.push(res_name.clone());
                                    }
                                }
                            }
                        }
                        "mut_resource" => {
                            if let Some(first) = args.first() {
                                if let ExprKind::Var(res_name) = &first.value.kind {
                                    if let Some(pos) = usage.resources.iter().position(|r| r == res_name) {
                                        usage.resources.remove(pos);
                                    }
                                    if !usage.mut_resources.contains(res_name) {
                                        usage.mut_resources.push(res_name.clone());
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            ExprKind::MethodCall(base, _method, _args) => {
                if let ExprKind::DotIdent(name) = &base.kind {
                    if name == "input" {
                        usage.has_input = true;
                    }
                }
            }
            _ => {}
        }
    };
    let mut body_clone = body.to_vec();
    Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body_clone);
    usage
}

impl Transpiler {
    pub(crate) fn emit_query(&mut self, q: &QueryDecl) {
        let fn_name = match (&q.self_type, &q.name) {
            (Some(st), Some(n)) => format!("{}_{}", camel_to_snake(st), camel_to_snake(n)),
            (Some(st), None) => format!("__boring_{}_query_{}_{}", camel_to_snake(st), q.line, q.col),
            (None, Some(n)) => camel_to_snake(n),
            (None, None) => format!("__boring_query_{}_{}", q.line, q.col),
        };
        let fn_name = escape_rust_keyword(&fn_name);

        // Scan ambient usage
        let usage = scan_ambient_usage(&q.body, &self.ecs_resource_vars);

        // Emit attribute comments if any
        for attr in &q.attrs {
            if attr.args.is_empty() {
                self.line(&format!("// @{}", attr.name));
            } else {
                self.line(&format!("// @{}({})", attr.name, attr.args.join(", ")));
            }
        }

        // Build parameters
        let mut params_rust = Vec::new();

        let has_query = q.self_type.is_some() || !q.params.is_empty() || q.where_clause.is_some() || usage.has_entity;
        if has_query {
            let mut comp_types = Vec::new();
            if usage.has_entity {
                comp_types.push("Entity".to_string());
            }

            // If this is a method query on a component, `self` is the first component
            if let Some(st) = &q.self_type {
                if q.is_req {
                    comp_types.push(format!("&{}", st));
                } else {
                    comp_types.push(format!("&mut {}", st));
                }
            }

            for p in &q.params {
                let ty_str = match &p.ty {
                    Some(t) => self.emit_type(t),
                    None => "()".to_string(),
                };
                if p.mutable {
                    comp_types.push(format!("&mut {}", ty_str));
                } else {
                    comp_types.push(format!("&{}", ty_str));
                }
            }
            if comp_types.is_empty() {
                comp_types.push("Entity".to_string());
            }

            let comp_str = if comp_types.len() == 1 && !usage.has_entity {
                comp_types[0].clone()
            } else {
                format!("({})", comp_types.join(", "))
            };

            let filter_str = if let Some(where_clause) = &q.where_clause {
                let mut filter_terms = Vec::new();
                for term in &where_clause.terms {
                    match term {
                        QueryFilterTerm::With(t) => filter_terms.push(format!("With<{}>", t)),
                        QueryFilterTerm::Without(t) => filter_terms.push(format!("Without<{}>", t)),
                        QueryFilterTerm::Changed(t) => filter_terms.push(format!("Changed<{}>", t)),
                        QueryFilterTerm::Added(t) => filter_terms.push(format!("Added<{}>", t)),
                    }
                }
                if filter_terms.is_empty() {
                    None
                } else if filter_terms.len() == 1 {
                    Some(filter_terms[0].clone())
                } else {
                    Some(format!("({})", filter_terms.join(", ")))
                }
            } else {
                None
            };

            let query_type = match filter_str {
                Some(f) => format!("Query<{}, {}>", comp_str, f),
                None => format!("Query<{}>", comp_str),
            };
            params_rust.push(format!("mut query: {}", query_type));
        }

        if usage.has_commands {
            params_rust.push("mut commands: Commands".to_string());
        }
        if usage.has_time {
            params_rust.push("time: Res<Time>".to_string());
        }
        if usage.has_input {
            params_rust.push("input: Res<ButtonInput<KeyCode>>".to_string());
        }
        for r in &usage.resources {
            let ty_name = if self.struct_fields.contains_key(r) {
                r.clone()
            } else {
                format!("Res_{}", r)
            };
            params_rust.push(format!("res_{}: Res<{}>", r.to_lowercase(), ty_name));
        }
        for r in &usage.mut_resources {
            let ty_name = if self.struct_fields.contains_key(r) {
                r.clone()
            } else {
                format!("Res_{}", r)
            };
            params_rust.push(format!("mut res_{}: ResMut<{}>", r.to_lowercase(), ty_name));
        }

        self.line(&format!("pub fn {}({}) {{", fn_name, params_rust.join(", ")));
        self.indent += 1;

        let prev_ecs = self.in_ecs_query;
        self.in_ecs_query = true;

        // Save local tracking state
        let prev_var_struct_types = std::mem::take(&mut self.var_struct_types);
        let prev_var_struct_type = std::mem::take(&mut self.var_struct_type);
        let prev_known_local_vars = self.known_local_vars.clone();

        if let Some(st) = &q.self_type {
            self.known_local_vars.insert("self_".to_string());
            self.known_local_vars.insert("self".to_string());
            self.var_struct_types.insert("self_".to_string(), st.clone());
            self.var_struct_types.insert("self".to_string(), st.clone());
            self.var_struct_type.insert("self_".to_string(), st.clone());
            self.var_struct_type.insert("self".to_string(), st.clone());
        }

        for p in &q.params {
            self.known_local_vars.insert(p.name.clone());
            if let Some(Type::Named(ref n)) = &p.ty {
                self.var_struct_types.insert(p.name.clone(), n.clone());
                self.var_struct_type.insert(p.name.clone(), n.clone());
            }
        }

        // Prepare body: if `self_type` is Some, rewrite `self` to `self_`
        let mut body = q.body.clone();
        if q.self_type.is_some() {
            let mut on_stmt = |_s: &mut Stmt| {};
            let mut on_expr = |e: &mut Expr| {
                if let ExprKind::Var(v) = &mut e.kind {
                    if v == "self" {
                        *v = "self_".to_string();
                    }
                }
            };
            Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body);
        }

        if has_query && (q.self_type.is_some() || !q.params.is_empty() || usage.has_entity) {
            let mut pat_bindings = Vec::new();
            if usage.has_entity {
                pat_bindings.push("__entity".to_string());
            }

            if q.self_type.is_some() {
                if q.is_req {
                    pat_bindings.push("self_".to_string());
                } else {
                    pat_bindings.push("mut self_".to_string());
                }
            }

            for p in &q.params {
                let name = escape_rust_keyword(&p.name);
                if p.mutable {
                    pat_bindings.push(format!("mut {}", name));
                } else {
                    pat_bindings.push(name);
                }
            }

            if pat_bindings.len() == 1 && !usage.has_entity {
                self.line(&format!("for {} in &mut query {{", pat_bindings[0]));
            } else {
                self.line(&format!("for ({}) in &mut query {{", pat_bindings.join(", ")));
            }
            self.indent += 1;

            for stmt in &body {
                self.emit_stmt(stmt, false);
            }

            self.indent -= 1;
            self.line("}");
        } else {
            // No loop (e.g. startup / command query without component params)
            for stmt in &body {
                self.emit_stmt(stmt, false);
            }
        }

        self.var_struct_types = prev_var_struct_types;
        self.var_struct_type = prev_var_struct_type;
        self.known_local_vars = prev_known_local_vars;
        self.in_ecs_query = prev_ecs;

        self.indent -= 1;
        self.line("}");
        self.blank();
    }
}
