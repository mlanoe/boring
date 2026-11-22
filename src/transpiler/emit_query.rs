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

fn format_filter_clause(where_clause: &Option<QueryWhereClause>) -> Option<String> {
    if let Some(clause) = where_clause {
        let mut filter_terms = Vec::new();
        for term in &clause.terms {
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
    }
}

fn ecs_comp_type_to_rust(ty: &Type) -> String {
    match ty {
        Type::Named(n) if n == "Entity" => "Entity".to_string(),
        Type::Mut(inner) => {
            match &**inner {
                Type::Optional(opt_inner) => {
                    let n = opt_inner.name().unwrap_or("()");
                    format!("Option<&mut {}>", n)
                }
                _ => {
                    let n = inner.name().unwrap_or("()");
                    format!("&mut {}", n)
                }
            }
        }
        Type::Optional(inner) => {
            match &**inner {
                Type::Mut(m) => {
                    let n = m.name().unwrap_or("()");
                    format!("Option<&mut {}>", n)
                }
                _ => {
                    let n = inner.name().unwrap_or("()");
                    format!("Option<&{}>", n)
                }
            }
        }
        _ => {
            let n = ty.name().unwrap_or("()");
            format!("&{}", n)
        }
    }
}

fn fetch_comp_to_rust(comp: &EcsFetchComponent) -> String {
    if comp.ty.name() == Some("Entity") {
        return "Entity".to_string();
    }
    let ty_name = comp.ty.name().unwrap_or("()");
    if comp.is_option {
        if comp.is_mut {
            format!("Option<&mut {}>", ty_name)
        } else {
            format!("Option<&{}>", ty_name)
        }
    } else if comp.is_mut {
        format!("&mut {}", ty_name)
    } else {
        format!("&{}", ty_name)
    }
}

pub(crate) struct CustomEcsScanner {
    pub(crate) fetches: Vec<(usize, usize, EcsFetchExpr)>,
    pub(crate) for_loops: Vec<(usize, usize, ForStmt)>,
}

pub(crate) fn scan_custom_ecs(body: &[Stmt]) -> CustomEcsScanner {
    let mut scanner = CustomEcsScanner {
        fetches: Vec::new(),
        for_loops: Vec::new(),
    };
    let mut on_expr = |e: &mut Expr| {
        if let ExprKind::EcsFetch(f) = &e.kind {
            scanner.fetches.push((e.line, e.col, (**f).clone()));
        }
    };
    let mut on_stmt = |s: &mut Stmt| {
        if let Stmt::For(for_stmt) = s {
            if for_stmt.where_clause.is_some() || for_stmt.var_types.iter().any(|t| t.is_some()) {
                scanner.for_loops.push((for_stmt.line, for_stmt.col, for_stmt.clone()));
            }
        }
    };
    let mut body_clone = body.to_vec();
    Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body_clone);
    scanner
}

fn is_standard_mutating_method(m: &str) -> bool {
    matches!(
        m,
        "insert"
            | "remove"
            | "clear"
            | "push"
            | "pop"
            | "retain"
            | "extend"
            | "sort"
            | "reverse"
            | "truncate"
            | "drain"
            | "rebuild"
            | "reset"
            | "update"
            | "set"
            | "add"
    )
}

fn canonical_res_name(name: &str, res_info_map: &std::collections::HashMap<String, ResourceInfo>) -> String {
    if let Some(info) = res_info_map.get(name) {
        info.rust_type.clone()
    } else {
        snake_to_pascal(&camel_to_snake(name))
    }
}

fn record_res_read(
    name: &str,
    usage: &mut AmbientUsage,
    res_info_map: &std::collections::HashMap<String, ResourceInfo>,
) {
    let canonical = canonical_res_name(name, res_info_map);
    if !usage.resources.contains(&canonical) && !usage.mut_resources.contains(&canonical) {
        usage.resources.push(canonical);
    }
}

fn record_res_mut(
    name: &str,
    usage: &mut AmbientUsage,
    res_info_map: &std::collections::HashMap<String, ResourceInfo>,
) {
    let canonical = canonical_res_name(name, res_info_map);
    if let Some(pos) = usage.resources.iter().position(|r| r == &canonical) {
        usage.resources.remove(pos);
    }
    if !usage.mut_resources.contains(&canonical) {
        usage.mut_resources.push(canonical);
    }
}

fn extract_ecs_context_resource_name(expr: &Expr) -> Option<String> {
    match &expr.kind {
        ExprKind::Field(base, field) => {
            if is_ecs_resource_obj(base) {
                return Some(field.clone());
            }
            extract_ecs_context_resource_name(base)
        }
        ExprKind::Index(base, _) | ExprKind::LabeledIndex(base, _) => extract_ecs_context_resource_name(base),
        _ => None,
    }
}

fn extract_resource_name(
    expr: &Expr,
    known_resources: &std::collections::HashSet<String>,
    res_info_map: &std::collections::HashMap<String, ResourceInfo>,
    local_vars: &std::collections::HashSet<String>,
) -> Option<String> {
    match &expr.kind {
        ExprKind::Var(name) => {
            if local_vars.contains(name) {
                return None;
            }
            if known_resources.contains(name) {
                if let Some(info) = res_info_map.get(name) {
                    if info.is_type && info.rust_type == *name {
                        return None;
                    }
                }
                Some(name.clone())
            } else {
                None
            }
        }
        ExprKind::DotIdent(name) => {
            if known_resources.contains(name) {
                if let Some(info) = res_info_map.get(name) {
                    if info.is_type && info.rust_type == *name {
                        return None;
                    }
                }
                Some(name.clone())
            } else {
                None
            }
        }
        ExprKind::Field(base, field) => {
            if is_resource_qualifier_obj(base) {
                return Some(field.clone());
            }
            if let ExprKind::DotIdent(d) = &base.kind {
                if d == "resource" {
                    return Some(field.clone());
                }
            }
            if is_ecs_resource_obj(base) {
                return Some(field.clone());
            }
            extract_resource_name(base, known_resources, res_info_map, local_vars)
        }
        ExprKind::Index(base, _) => extract_resource_name(base, known_resources, res_info_map, local_vars),
        _ => None,
    }
}

fn scan_ambient_usage(
    body: &[Stmt],
    known_resources: &std::collections::HashSet<String>,
    res_info_map: &std::collections::HashMap<String, ResourceInfo>,
    mutating_methods: &std::collections::HashSet<String>,
    fn_mutable: &std::collections::HashMap<String, Vec<bool>>,
    fn_ecs_usage: &std::collections::HashMap<String, FnEcsUsage>,
    local_vars: &std::collections::HashSet<String>,
) -> AmbientUsage {
    let mut usage = AmbientUsage::default();
    let mut on_stmt = |_s: &mut Stmt| {};
    let mut on_expr = |e: &mut Expr| {
        match &e.kind {
            ExprKind::DotIdent(name) => {
                match name.as_str() {
                    "time" => usage.has_time = true,
                    "input" => usage.has_input = true,
                    "entity" => usage.has_entity = true,
                    "commands" => usage.has_commands = true,
                    _ => {
                        if known_resources.contains(name) {
                            record_res_read(name, &mut usage, res_info_map);
                        }
                    }
                }
            }
            ExprKind::Field(base, field) => {
                if is_ecs_qualifier_obj(base) && field == "entity" {
                    usage.has_entity = true;
                } else if is_time_qualifier_obj(base) {
                    usage.has_time = true;
                } else if is_input_qualifier_obj(base) {
                    usage.has_input = true;
                } else if is_commands_qualifier_obj(base) {
                    usage.has_commands = true;
                } else if is_resource_qualifier_obj(base) {
                    record_res_read(field, &mut usage, res_info_map);
                } else if let ExprKind::DotIdent(name) = &base.kind {
                    match name.as_str() {
                        "time" => usage.has_time = true,
                        "input" => usage.has_input = true,
                        "resource" => {
                            record_res_read(field, &mut usage, res_info_map);
                        }
                        _ => {}
                    }
                } else if is_ecs_resource_obj(base) {
                    record_res_read(field, &mut usage, res_info_map);
                }
            }
            ExprKind::Var(name) => {
                if local_vars.contains(name) {
                    return;
                }
                if name == "entity" {
                    usage.has_entity = true;
                } else if name == "commands" {
                    usage.has_commands = true;
                } else if name == "time" {
                    usage.has_time = true;
                } else if name == "input" {
                    usage.has_input = true;
                } else if known_resources.contains(name) {
                    let is_type_name = res_info_map.get(name).map(|i| i.is_type && i.rust_type == *name).unwrap_or(false);
                    if !is_type_name {
                        record_res_read(name, &mut usage, res_info_map);
                    }
                }
            }
            ExprKind::Assign(target, _) => {
                let mut mutated_res: Option<String> = None;
                match &target.kind {
                    ExprKind::Var(n) if !local_vars.contains(n) && known_resources.contains(n) => {
                        mutated_res = Some(n.clone());
                    }
                    ExprKind::DotIdent(n) if known_resources.contains(n) => {
                        mutated_res = Some(n.clone());
                    }
                    ExprKind::Index(base, _) => {
                        if let Some(r) = extract_resource_name(base, known_resources, res_info_map, local_vars) {
                            mutated_res = Some(r);
                        }
                    }
                    ExprKind::Field(base, field) => {
                        if is_resource_qualifier_obj(base) {
                            mutated_res = Some(field.clone());
                        } else if let ExprKind::DotIdent(d) = &base.kind {
                            if d == "resource" {
                                mutated_res = Some(field.clone());
                            } else if known_resources.contains(d) {
                                mutated_res = Some(d.clone());
                            }
                        } else if is_ecs_resource_obj(base) {
                            mutated_res = Some(field.clone());
                        } else if let ExprKind::Var(v) = &base.kind {
                            if known_resources.contains(v) {
                                mutated_res = Some(v.clone());
                            }
                        } else if let ExprKind::Field(inner, inner_field) = &base.kind {
                            if is_resource_qualifier_obj(inner) {
                                mutated_res = Some(inner_field.clone());
                            } else if let ExprKind::DotIdent(d) = &inner.kind {
                                if d == "resource" {
                                    mutated_res = Some(inner_field.clone());
                                }
                            } else if is_ecs_resource_obj(inner) {
                                mutated_res = Some(inner_field.clone());
                            }
                        }
                    }
                    _ => {}
                }
                if let Some(r) = mutated_res {
                    record_res_mut(&r, &mut usage, res_info_map);
                }
            }
            ExprKind::Call(callee, args) => {
                let is_despawn = match &callee.kind {
                    ExprKind::DotIdent(name) => name == "despawn",
                    ExprKind::Var(name) if !local_vars.contains(name) => name == "despawn",
                    ExprKind::Field(base, field) => field == "despawn" && is_ecs_qualifier_obj(base),
                    _ => false,
                };
                if is_despawn {
                    usage.has_commands = true;
                    usage.has_entity = true;
                }
                let is_spawn = match &callee.kind {
                    ExprKind::DotIdent(name) => name == "spawn",
                    ExprKind::Var(name) if !local_vars.contains(name) => name == "spawn",
                    ExprKind::Field(base, field) => field == "spawn" && is_ecs_qualifier_obj(base),
                    _ => false,
                };
                if is_spawn {
                    usage.has_commands = true;
                }
                if let ExprKind::DotIdent(name) = &callee.kind {
                    match name.as_str() {
                        "resource" => {
                            if let Some(first) = args.first() {
                                if let ExprKind::Var(res_name) = &first.value.kind {
                                    record_res_read(res_name, &mut usage, res_info_map);
                                }
                            }
                        }
                        "mut_resource" => {
                            if let Some(first) = args.first() {
                                if let ExprKind::Var(res_name) = &first.value.kind {
                                    record_res_mut(res_name, &mut usage, res_info_map);
                                }
                            }
                        }
                        _ => {}
                    }
                } else if let ExprKind::Var(fn_name) = &callee.kind {
                    if let Some(fn_u) = fn_ecs_usage.get(fn_name) {
                        for r in &fn_u.reads {
                            record_res_read(r, &mut usage, res_info_map);
                        }
                        for w in &fn_u.writes {
                            record_res_mut(w, &mut usage, res_info_map);
                        }
                    }
                    let mut_flags = fn_mutable.get(fn_name);
                    for (i, arg) in args.iter().enumerate() {
                        if let Some(res_name) = extract_resource_name(&arg.value, known_resources, res_info_map, local_vars) {
                            let is_mut = mut_flags.and_then(|f| f.get(i)).copied().unwrap_or(false);
                            if is_mut {
                                record_res_mut(&res_name, &mut usage, res_info_map);
                            } else {
                                record_res_read(&res_name, &mut usage, res_info_map);
                            }
                        }
                    }
                }
            }
            ExprKind::Pipe(_lhs, name, _args) => {
                if let Some(fn_u) = fn_ecs_usage.get(name) {
                    for r in &fn_u.reads {
                        record_res_read(r, &mut usage, res_info_map);
                    }
                    for w in &fn_u.writes {
                        record_res_mut(w, &mut usage, res_info_map);
                    }
                }
            }
            ExprKind::MethodCall(base, method, _args) => {
                if is_ecs_qualifier_obj(base) {
                    if method == "despawn" {
                        usage.has_commands = true;
                        usage.has_entity = true;
                    } else if method == "spawn" {
                        usage.has_commands = true;
                    }
                } else if is_time_qualifier_obj(base) {
                    usage.has_time = true;
                } else if is_input_qualifier_obj(base) {
                    usage.has_input = true;
                } else if is_commands_qualifier_obj(base) {
                    usage.has_commands = true;
                } else if let ExprKind::DotIdent(name) = &base.kind {
                    if name == "input" {
                        usage.has_input = true;
                    } else if name == "commands" {
                        usage.has_commands = true;
                    } else if known_resources.contains(name) {
                        if is_standard_mutating_method(method)
                            || mutating_methods.contains(method)
                            || mutating_methods.contains(&format!("{}::{}", name, method))
                        {
                            record_res_mut(name, &mut usage, res_info_map);
                        } else {
                            record_res_read(name, &mut usage, res_info_map);
                        }
                    }
                } else if let Some(res_name) = extract_resource_name(base, known_resources, res_info_map, local_vars) {
                    let is_mut = is_standard_mutating_method(method)
                        || mutating_methods.contains(method)
                        || mutating_methods.contains(&format!("{}::{}", res_name, method))
                        || mutating_methods.contains(&format!(
                            "{}::{}",
                            canonical_res_name(&res_name, res_info_map),
                            method
                        ));
                    if is_mut {
                        record_res_mut(&res_name, &mut usage, res_info_map);
                    } else {
                        record_res_read(&res_name, &mut usage, res_info_map);
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
    fn is_bevy_system_param(&self, p: &Param) -> bool {
        if p.name == "commands" {
            return true;
        }
        if let Some(t) = &p.ty {
            if self.is_resource_type(t) {
                return true;
            }
            match t {
                Type::Generic(name, _) => {
                    if matches!(
                        name.as_str(),
                        "Res"
                            | "ResMut"
                            | "Single"
                            | "Local"
                            | "On"
                            | "EventReader"
                            | "EventWriter"
                            | "MessageReader"
                            | "MessageWriter"
                            | "Query"
                            | "Commands"
                    ) {
                        return true;
                    }
                }
                Type::Named(name) => {
                    if matches!(name.as_str(), "Commands" | "TextUiWriter" | "Time") {
                        return true;
                    }
                    if name.starts_with("Res<")
                        || name.starts_with("ResMut<")
                        || name.starts_with("Single<")
                        || name.starts_with("Local<")
                        || name.starts_with("On<")
                        || name.starts_with("Query<")
                        || name.starts_with("Commands<")
                        || name.starts_with("EventReader<")
                        || name.starts_with("EventWriter<")
                    {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    #[allow(dead_code)]
    pub(crate) fn is_component_type(&self, ty: &Type) -> bool {
        match ty {
            Type::Named(name) => {
                self.ecs_component_types.contains(name) || matches!(
                    name.as_str(),
                    "Transform" | "GlobalTransform" | "Sprite" | "Visibility" | "InheritedVisibility" | "ViewVisibility" | "Camera" | "Camera2d" | "Text2d" | "Anchor"
                )
            }
            Type::Generic(name, _) => {
                matches!(name.as_str(), "Handle" | "MeshMaterial2d")
            }
            _ => false,
        }
    }

    pub(crate) fn is_ecs_fn(&self, f: &FnDecl) -> bool {
        if let Some(ret) = &f.return_ty {
            if !matches!(ret, Type::Void | Type::Nil) {
                return false;
            }
        }
        if f.ecs_where_clause.is_some() {
            return true;
        }
        if f.params.first().and_then(|p| p.ty.as_ref()).map(|ty| {
            match ty {
                Type::Generic(g, _) => g == "On" || g == "Trigger",
                Type::Named(n) => n.starts_with("On<") || n.starts_with("Trigger<"),
                _ => false,
            }
        }).unwrap_or(false) {
            return true;
        }
        let custom = scan_custom_ecs(&f.body);
        if !custom.fetches.is_empty() || !custom.for_loops.is_empty() {
            return true;
        }
        if f.params.iter().any(|p| p.ty.as_ref().map(|t| self.is_component_type(t)).unwrap_or(false)) {
            return true;
        }
        let has_system_attr = f.attrs.iter().any(|a| {
            matches!(
                a.name.to_ascii_lowercase().as_str(),
                "schedule" | "startup" | "update" | "fixedupdate" | "fixed_update" | "observer" | "observe" | "after" | "before" | "chain" | "on_event"
            )
        });
        if has_system_attr {
            return true;
        }

        false
    }

    pub(crate) fn emit_ecs_fn(&mut self, f: &FnDecl, self_type: Option<&str>) {
        let fn_name = match (self_type, f.name.as_str()) {
            (Some(st), n) => format!("{}_{}", camel_to_snake(st), camel_to_snake(n)),
            (None, n) => camel_to_snake(n),
        };
        let fn_name = escape_rust_keyword(&fn_name);

        // Separate system params from entity component query params
        let mut system_params = Vec::new();
        let mut component_params = Vec::new();
        for p in &f.params {
            if self.is_bevy_system_param(p) {
                system_params.push(p);
            } else {
                component_params.push(p);
            }
        }

        let mut local_vars = std::collections::HashSet::new();
        for p in &f.params {
            local_vars.insert(p.name.clone());
        }
        crate::transpiler::helpers::collect_local_decl_names(&f.body, &mut local_vars);

        // Scan ambient usage
        let mut usage = scan_ambient_usage(
            &f.body,
            &self.ecs_resource_vars,
            &self.ecs_resource_info,
            &self.ecs_resource_mutating_methods,
            &self.fn_mutable,
            &self.fn_ecs_usage,
            &local_vars,
        );

        let query_needs_ecs_context = {
            let mut needs = false;
            let mut on_callee = |e: &mut Expr| {
                if let ExprKind::Call(callee, _) = &e.kind {
                    if let ExprKind::Var(callee_name) = &callee.kind {
                        if self.fns_needing_ecs_context.contains(callee_name) {
                            needs = true;
                        }
                    }
                } else if let ExprKind::Pipe(_, name, _) = &e.kind {
                    if self.fns_needing_ecs_context.contains(name) {
                        needs = true;
                    }
                }
            };
            let mut on_stmt = |_s: &mut Stmt| {};
            let mut body_clone = f.body.clone();
            Walker { expr: &mut on_callee, stmt: &mut on_stmt }.stmts(&mut body_clone);
            needs
        };

        if query_needs_ecs_context {
            for r in &self.ecs_context_resource_types {
                if self.ecs_context_mut_resource_types.contains(r) {
                    record_res_mut(r, &mut usage, &self.ecs_resource_info);
                } else {
                    record_res_read(r, &mut usage, &self.ecs_resource_info);
                }
            }
        }

        // Emit attribute comments if any
        for attr in &f.attrs {
            if attr.args.is_empty() {
                self.line(&format!("// @{}", attr.name));
            } else {
                self.line(&format!("// @{}({})", attr.name, attr.args.join(", ")));
            }
        }

        self.ecs_fetch_params.borrow_mut().clear();
        self.ecs_for_query_params.borrow_mut().clear();
        self.ecs_query_throws = f.throws;

        let custom_ecs = scan_custom_ecs(&f.body);
        let is_custom_query = !custom_ecs.fetches.is_empty() || !custom_ecs.for_loops.is_empty();

        let mut used_param_names = std::collections::HashSet::new();
        for p in &f.params {
            used_param_names.insert(p.name.clone());
        }

        // Build parameters
        let mut params_rust = Vec::new();

        for (fetch_line, fetch_col, fetch) in &custom_ecs.fetches {
            let comp_strs: Vec<String> = fetch.components.iter().map(fetch_comp_to_rust).collect();
            let comp_str = if comp_strs.is_empty() {
                "Entity".to_string()
            } else if comp_strs.len() == 1 {
                comp_strs[0].clone()
            } else {
                format!("({})", comp_strs.join(", "))
            };
            let filter_str = format_filter_clause(&fetch.where_clause);
            let any_mut = fetch.components.iter().any(|c| c.is_mut);

            let (is_single, base_type) = if fetch.kind == EcsFetchKind::Single && !fetch.is_optional && !f.throws {
                (true, match &filter_str {
                    Some(f) => format!("Single<{}, {}>", comp_str, f),
                    None => format!("Single<{}>", comp_str),
                })
            } else {
                (false, match &filter_str {
                    Some(f) => format!("Query<{}, {}>", comp_str, f),
                    None => format!("Query<{}>", comp_str),
                })
            };

            let base_name = if let Some(where_c) = &fetch.where_clause {
                where_c.terms.first().map(|t| match t {
                    QueryFilterTerm::With(w) => camel_to_snake(w),
                    QueryFilterTerm::Without(w) => camel_to_snake(w),
                    QueryFilterTerm::Changed(w) => camel_to_snake(w),
                    QueryFilterTerm::Added(w) => camel_to_snake(w),
                }).unwrap_or_else(|| "ecs".to_string())
            } else if let Some(first) = fetch.components.first() {
                first.ty.name().map(camel_to_snake).unwrap_or_else(|| "ecs".to_string())
            } else {
                "ecs".to_string()
            };
            let suffix = if is_single { "single" } else { "query" };
            let mut candidate = format!("{}_{}", base_name, suffix);
            let mut counter = 1;
            while used_param_names.contains(&candidate) {
                candidate = format!("{}_{}_{}", base_name, suffix, counter);
                counter += 1;
            }
            used_param_names.insert(candidate.clone());
            self.ecs_fetch_params.borrow_mut().insert((*fetch_line, *fetch_col), candidate.clone());
            let mut_prefix = if any_mut { "mut " } else { "" };
            params_rust.push(format!("{}{}: {}", mut_prefix, candidate, base_type));
        }

        for (for_line, for_col, for_loop) in &custom_ecs.for_loops {
            let comp_strs: Vec<String> = for_loop.vars.iter().zip(for_loop.var_types.iter()).map(|(v, ty)| {
                if v == "entity" {
                    "Entity".to_string()
                } else if let Some(t) = ty {
                    ecs_comp_type_to_rust(t)
                } else {
                    format!("&{}", v)
                }
            }).collect();
            let comp_str = if comp_strs.is_empty() {
                "Entity".to_string()
            } else if comp_strs.len() == 1 {
                comp_strs[0].clone()
            } else {
                format!("({})", comp_strs.join(", "))
            };
            let filter_str = format_filter_clause(&for_loop.where_clause);
            let query_type = match &filter_str {
                Some(f) => format!("Query<{}, {}>", comp_str, f),
                None => format!("Query<{}>", comp_str),
            };
            let any_mut = for_loop.var_types.iter().any(|t| matches!(t, Some(Type::Mut(_))));
            let base_name = if let Some(where_c) = &for_loop.where_clause {
                where_c.terms.first().map(|t| match t {
                    QueryFilterTerm::With(w) => camel_to_snake(w),
                    QueryFilterTerm::Without(w) => camel_to_snake(w),
                    QueryFilterTerm::Changed(w) => camel_to_snake(w),
                    QueryFilterTerm::Added(w) => camel_to_snake(w),
                }).unwrap_or_else(|| "items".to_string())
            } else if let Some(Some(t)) = for_loop.var_types.iter().find(|t| t.is_some() && t.as_ref().unwrap().name() != Some("Entity")) {
                t.name().map(camel_to_snake).unwrap_or_else(|| "items".to_string())
            } else {
                "items".to_string()
            };
            let mut candidate = format!("{}_query", base_name);
            let mut counter = 1;
            while used_param_names.contains(&candidate) {
                candidate = format!("{}_query_{}", base_name, counter);
                counter += 1;
            }
            used_param_names.insert(candidate.clone());
            self.ecs_for_query_params.borrow_mut().insert((*for_line, *for_col), candidate.clone());
            let mut_prefix = if any_mut { "mut " } else { "" };
            params_rust.push(format!("{}{}: {}", mut_prefix, candidate, query_type));
        }

        let has_query = !is_custom_query && (self_type.is_some() || !component_params.is_empty() || f.ecs_where_clause.is_some() || usage.has_entity);
        if has_query {
            let mut comp_types = Vec::new();
            if usage.has_entity {
                comp_types.push("Entity".to_string());
            }

            // If this is a method query on a component, `self` is the first component
            if let Some(st) = self_type {
                if !f.mutating {
                    comp_types.push(format!("&{}", st));
                } else {
                    comp_types.push(format!("&mut {}", st));
                }
            }

            for p in &component_params {
                let ty_str = match &p.ty {
                    Some(t) => self.emit_type(t),
                    None => "()".to_string(),
                };
                let comp_ty = if ty_str.starts_with("Option<") && ty_str.ends_with('>') {
                    let inner = &ty_str[7..ty_str.len() - 1];
                    if p.mutable {
                        format!("Option<&mut {}>", inner)
                    } else {
                        format!("Option<&{}>", inner)
                    }
                } else if p.mutable {
                    format!("&mut {}", ty_str)
                } else {
                    format!("&{}", ty_str)
                };
                comp_types.push(comp_ty);
            }
            if comp_types.is_empty() {
                comp_types.push("Entity".to_string());
            }

            let comp_str = if comp_types.len() == 1 && !usage.has_entity {
                comp_types[0].clone()
            } else {
                format!("({})", comp_types.join(", "))
            };

            let filter_str = if let Some(where_clause) = &f.ecs_where_clause {
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

        // Track what explicit system parameters are already present
        let mut has_explicit_commands = false;
        let mut has_explicit_time = false;
        let mut has_explicit_input = false;
        let mut explicit_resources = std::collections::HashSet::new();

        for p in &system_params {
            let mut ty_str = if p.name == "commands" {
                "Commands".to_string()
            } else {
                match &p.ty {
                    Some(t) => self.emit_type(t),
                    None => "()".to_string(),
                }
            };
            if ty_str == "Commands" || ty_str.starts_with("Commands<") {
                has_explicit_commands = true;
            }
            if ty_str.contains("Time") {
                has_explicit_time = true;
            }
            if ty_str.contains("ButtonInput") || ty_str.contains("KeyCode") {
                has_explicit_input = true;
            }

            let is_already_res = ty_str.starts_with("Res<") || ty_str.starts_with("ResMut<");
            let is_special_system_param = ty_str == "Commands"
                || ty_str.starts_with("Commands<")
                || ty_str == "TextUiWriter"
                || ty_str.starts_with("Single<")
                || ty_str.starts_with("Query<")
                || ty_str.starts_with("On<")
                || ty_str.starts_with("Local<")
                || ty_str.starts_with("EventReader<")
                || ty_str.starts_with("EventWriter<")
                || ty_str.starts_with("MessageReader<")
                || ty_str.starts_with("MessageWriter<");

            if !is_already_res && !is_special_system_param {
                if let Some(t) = &p.ty {
                    if self.is_resource_type(t) {
                        if p.mutable {
                            ty_str = format!("ResMut<{}>", ty_str);
                        } else {
                            ty_str = format!("Res<{}>", ty_str);
                        }
                    }
                }
            }

            if let Some(Type::Generic(g, args)) = &p.ty {
                if (g == "Res" || g == "ResMut") && !args.is_empty() {
                    let inner_s = self.emit_type(&args[0]);
                    explicit_resources.insert(inner_s.clone());
                    explicit_resources.insert(camel_to_snake(&inner_s));
                }
            }
            if (ty_str.starts_with("Res<") || ty_str.starts_with("ResMut<")) && ty_str.ends_with('>') {
                let start = ty_str.find('<').unwrap() + 1;
                let end = ty_str.rfind('>').unwrap();
                let inner = &ty_str[start..end];
                explicit_resources.insert(inner.to_string());
                explicit_resources.insert(camel_to_snake(inner));
            }

            let is_mut = p.mutable || ty_str == "Commands" || ty_str.starts_with("ResMut<");
            let param_name = escape_rust_keyword(&p.name);
            if is_mut {
                params_rust.push(format!("mut {}: {}", param_name, ty_str));
            } else {
                params_rust.push(format!("{}: {}", param_name, ty_str));
            }
        }

        if usage.has_commands && !has_explicit_commands {
            params_rust.push("mut commands: Commands".to_string());
        }
        if usage.has_time && !has_explicit_time {
            params_rust.push("time: Res<Time>".to_string());
        }
        if usage.has_input && !has_explicit_input {
            params_rust.push("input: Res<ButtonInput<KeyCode>>".to_string());
        }

        for r in &usage.resources {
            let (ty_name, param_name) = match self.ecs_resource_info.get(r) {
                Some(info) => (info.rust_type.clone(), info.param_name.clone()),
                None => (r.clone(), format!("res_{}", camel_to_snake(r))),
            };
            if !explicit_resources.contains(&ty_name)
                && !explicit_resources.contains(&param_name)
                && !usage.mut_resources.contains(r)
            {
                params_rust.push(format!("{}: Res<{}>", param_name, ty_name));
            }
        }
        for r in &usage.mut_resources {
            let (ty_name, param_name) = match self.ecs_resource_info.get(r) {
                Some(info) => (info.rust_type.clone(), info.param_name.clone()),
                None => (r.clone(), format!("res_{}", camel_to_snake(r))),
            };
            if !explicit_resources.contains(&ty_name) && !explicit_resources.contains(&param_name) {
                params_rust.push(format!("mut {}: ResMut<{}>", param_name, ty_name));
            }
        }

        let param_decl = if params_rust.len() > 15 {
            // Bevy function systems support at most 16 parameters.
            // When there are more than 15 parameters, group them into tuples of at most 12 parameters.
            let chunks: Vec<_> = params_rust.chunks(12).collect();
            let tuple_params: Vec<String> = chunks
                .iter()
                .map(|chunk| {
                    if chunk.len() == 1 {
                        chunk[0].clone()
                    } else {
                        let mut pats = Vec::new();
                        let mut tys = Vec::new();
                        for p in *chunk {
                            if let Some((pat, ty)) = p.split_once(": ") {
                                pats.push(pat.trim().to_string());
                                tys.push(ty.trim().to_string());
                            } else {
                                pats.push(p.clone());
                                tys.push(p.clone());
                            }
                        }
                        format!("({}): ({})", pats.join(", "), tys.join(", "))
                    }
                })
                .collect();
            tuple_params.join(", ")
        } else {
            params_rust.join(", ")
        };

        let ret_clause = if f.throws {
            " -> Result<(), Box<dyn std::error::Error + Send + Sync>>"
        } else {
            ""
        };
        self.line(&format!("pub fn {}({}){} {{", fn_name, param_decl, ret_clause));
        self.indent += 1;

        let prev_ecs = self.in_ecs_query;
        self.in_ecs_query = true;

        // Save local tracking state
        let prev_var_struct_types = std::mem::take(&mut self.var_struct_types);
        let prev_var_struct_type = std::mem::take(&mut self.var_struct_type);
        let prev_known_local_vars = self.known_local_vars.clone();

        let prev_fn_current_params = std::mem::take(&mut self.fn_current_params);
        let prev_fn_current_params_mut = std::mem::take(&mut self.fn_current_params_mut);
        let prev_fn_current_param_lines = std::mem::take(&mut self.fn_current_param_lines);
        let prev_fn_current_param_cols = std::mem::take(&mut self.fn_current_param_cols);
        let prev_fn_current_params_rebind_only = std::mem::take(&mut self.fn_current_params_rebind_only);

        self.fn_current_params = f.params.iter()
            .filter_map(|p| p.ty.as_ref().map(|ty| (p.name.clone(), ty.clone())))
            .collect();
        self.fn_current_param_lines = f.params.iter()
            .map(|p| (p.name.clone(), p.line))
            .collect();
        self.fn_current_param_cols = f.params.iter()
            .map(|p| (p.name.clone(), p.col))
            .collect();
        self.fn_current_params_mut = f.params.iter()
            .filter(|p| p.mutable)
            .map(|p| p.name.clone())
            .collect();

        if let Some(st) = self_type {
            self.known_local_vars.insert("self_".to_string());
            self.known_local_vars.insert("self".to_string());
            self.var_struct_types.insert("self_".to_string(), st.to_string());
            self.var_struct_types.insert("self".to_string(), st.to_string());
            self.var_struct_type.insert("self_".to_string(), st.to_string());
            self.var_struct_type.insert("self".to_string(), st.to_string());
        }

        for p in &f.params {
            self.known_local_vars.insert(p.name.clone());
            if let Some(Type::Named(ref n)) = &p.ty {
                self.var_struct_types.insert(p.name.clone(), n.clone());
                self.var_struct_type.insert(p.name.clone(), n.clone());
            }
            if let Some(Type::Generic(ref g, ref args)) = &p.ty {
                if (g == "Res" || g == "ResMut") && !args.is_empty() {
                    if let Type::Named(ref n) = args[0] {
                        self.var_struct_types.insert(p.name.clone(), n.clone());
                        self.var_struct_type.insert(p.name.clone(), n.clone());
                    }
                }
            }
        }

        // Prepare body: if `self_type` is Some, rewrite `self` to `self_`
        let mut body = f.body.clone();
        if self_type.is_some() {
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

        if is_custom_query {
            for stmt in &body {
                self.emit_stmt(stmt, false);
            }
            if f.throws {
                self.line("Ok(())");
            }
        } else if has_query && (self_type.is_some() || !component_params.is_empty() || usage.has_entity) {
            let mut pat_bindings = Vec::new();
            if usage.has_entity {
                pat_bindings.push("__entity".to_string());
            }

            if self_type.is_some() {
                if !f.mutating {
                    pat_bindings.push("self_".to_string());
                } else {
                    pat_bindings.push("mut self_".to_string());
                }
            }

            for p in &component_params {
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
            if f.throws {
                self.line("Ok(())");
            }
        } else {
            // No loop (e.g. system without component params, or pure resource/commands system)
            for stmt in &body {
                self.emit_stmt(stmt, false);
            }
            if f.throws {
                self.line("Ok(())");
            }
        }

        self.var_struct_types = prev_var_struct_types;
        self.var_struct_type = prev_var_struct_type;
        self.known_local_vars = prev_known_local_vars;
        self.in_ecs_query = prev_ecs;
        self.ecs_query_throws = false;
        self.fn_current_params = prev_fn_current_params;
        self.fn_current_params_mut = prev_fn_current_params_mut;
        self.fn_current_param_lines = prev_fn_current_param_lines;
        self.fn_current_param_cols = prev_fn_current_param_cols;
        self.fn_current_params_rebind_only = prev_fn_current_params_rebind_only;

        self.indent -= 1;
        self.line("}");
        self.blank();
    }

    pub(crate) fn analyze_ecs_context_usage(&mut self) {
        let mut fn_direct_usage: std::collections::HashMap<String, FnEcsUsage> = std::collections::HashMap::new();
        let mut fn_calls: std::collections::HashMap<String, std::collections::HashSet<String>> = std::collections::HashMap::new();
        let mut fns_declaring_ecs: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Collect all items from all deep-scanned programs
        let mut all_items: Vec<crate::ast::Item> = Vec::new();
        for prog in &self.deep_scanned_programs {
            for item in &prog.items {
                all_items.push(item.clone());
            }
        }

        for item in &all_items {
            if let Item::Fn(f) = item {
                let mut usage = FnEcsUsage::default();
                let mut calls = std::collections::HashSet::new();
                let mut has_local_ecs = false;

                let mut on_stmt = |s: &mut Stmt| {
                    if let Stmt::Let(l) = s {
                        if l.name == "ecs" {
                            has_local_ecs = true;
                        }
                    }
                };
                let mut on_expr = |e: &mut Expr| {
                    match &e.kind {
                        ExprKind::Field(base, field) if is_ecs_resource_obj(base) => {
                            let canonical = canonical_res_name(field, &self.ecs_resource_info);
                            usage.reads.insert(canonical);
                        }
                        ExprKind::Assign(target, _) | ExprKind::QuestionAssign(target, _) => {
                            let mut mutated_res: Option<String> = None;
                            match &target.kind {
                                ExprKind::Index(base, _) | ExprKind::LabeledIndex(base, _) => {
                                    if let Some(r) = extract_ecs_context_resource_name(base) {
                                        mutated_res = Some(r);
                                    }
                                }
                                ExprKind::Field(base, field) => {
                                    if is_ecs_resource_obj(base) {
                                        mutated_res = Some(field.clone());
                                    } else if let ExprKind::Field(inner, inner_field) = &base.kind {
                                        if is_ecs_resource_obj(inner) {
                                            mutated_res = Some(inner_field.clone());
                                        }
                                    }
                                }
                                _ => {}
                            }
                            if let Some(r) = mutated_res {
                                let canonical = canonical_res_name(&r, &self.ecs_resource_info);
                                usage.writes.insert(canonical.clone());
                                usage.reads.insert(canonical);
                            }
                        }
                        ExprKind::Call(callee, args) => {
                            if let ExprKind::Var(callee_name) = &callee.kind {
                                calls.insert(callee_name.clone());
                                let mut_flags = self.fn_mutable.get(callee_name);
                                for (i, arg) in args.iter().enumerate() {
                                    if let Some(res_name) = extract_ecs_context_resource_name(&arg.value) {
                                        let canonical = canonical_res_name(&res_name, &self.ecs_resource_info);
                                        let is_mut = mut_flags.and_then(|f| f.get(i)).copied().unwrap_or(false);
                                        if is_mut {
                                            usage.writes.insert(canonical.clone());
                                        }
                                        usage.reads.insert(canonical);
                                    }
                                }
                            }
                        }
                        ExprKind::Pipe(lhs, name, _args) => {
                            calls.insert(name.clone());
                            if let Some(res_name) = extract_ecs_context_resource_name(lhs) {
                                let canonical = canonical_res_name(&res_name, &self.ecs_resource_info);
                                let mut_flags = self.fn_mutable.get(name);
                                let is_mut = mut_flags.and_then(|f| f.first()).copied().unwrap_or(false);
                                if is_mut {
                                    usage.writes.insert(canonical.clone());
                                }
                                usage.reads.insert(canonical);
                            }
                        }
                        ExprKind::MethodCall(base, method, _args) => {
                            if let Some(res_name) = extract_ecs_context_resource_name(base) {
                                let canonical = canonical_res_name(&res_name, &self.ecs_resource_info);
                                let is_mut = is_standard_mutating_method(method)
                                    || self.ecs_resource_mutating_methods.contains(method)
                                    || self.ecs_resource_mutating_methods.contains(&format!("{}::{}", res_name, method))
                                    || self.ecs_resource_mutating_methods.contains(&format!("{}::{}", canonical, method));
                                if is_mut {
                                    usage.writes.insert(canonical.clone());
                                }
                                usage.reads.insert(canonical);
                            }
                        }
                        _ => {}
                    }
                };

                let mut body_clone = f.body.clone();
                Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body_clone);

                if has_local_ecs {
                    fns_declaring_ecs.insert(f.name.clone());
                }

                fn_direct_usage.insert(f.name.clone(), usage);
                fn_calls.insert(f.name.clone(), calls);
            }
        }

        // Fixed-point propagation of usage across call graph
        let mut all_usage = fn_direct_usage;
        let mut changed = true;
        while changed {
            changed = false;
            for (caller, callees) in &fn_calls {
                for callee in callees {
                    if fns_declaring_ecs.contains(callee) {
                        continue;
                    }
                    if let Some(callee_usage) = all_usage.get(callee).cloned() {
                        let caller_usage = all_usage.entry(caller.clone()).or_default();
                        for r in callee_usage.reads {
                            if caller_usage.reads.insert(r) {
                                changed = true;
                            }
                        }
                        for w in callee_usage.writes {
                            if caller_usage.writes.insert(w) {
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        self.fn_ecs_usage = all_usage.clone();

        let mut fns_needing = std::collections::HashSet::new();
        let mut res_types_set = std::collections::HashSet::new();
        let mut mut_res_types_set = std::collections::HashSet::new();

        for (fn_name, usage) in &all_usage {
            if !usage.reads.is_empty() || !usage.writes.is_empty() {
                fns_needing.insert(fn_name.clone());
                for r in &usage.reads {
                    res_types_set.insert(r.clone());
                }
                for w in &usage.writes {
                    mut_res_types_set.insert(w.clone());
                }
            }
        }

        for decl_fn in &fns_declaring_ecs {
            fns_needing.remove(decl_fn);
        }
        for item in &all_items {
            if let Item::Fn(f) = item {
                if self.is_ecs_fn(f) {
                    fns_needing.remove(&f.name);
                }
            }
        }

        self.fns_needing_ecs_context = fns_needing;
        let mut res_types: Vec<String> = res_types_set.into_iter().collect();
        res_types.sort();
        self.ecs_context_resource_types = res_types;
        self.ecs_context_mut_resource_types = mut_res_types_set;
    }
}
