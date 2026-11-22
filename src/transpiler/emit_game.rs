use crate::ast::*;
use crate::ast_walk::Walker;
use super::Transpiler;
use super::escape_str;
use super::helpers::{camel_to_snake, escape_rust_keyword};

struct ScheduledSystem {
    rendered: String,
    chain: bool,
}

fn collect_all_items<'a>(items: &'a [Item], out: &mut Vec<&'a Item>) {
    for item in items {
        out.push(item);
        if let Item::Mod(m) = item {
            collect_all_items(&m.items, out);
        }
    }
}

impl Transpiler {
    pub(crate) fn emit_game_main(&mut self, program: &Program, game: &StructDecl, extra_stmts: &[&Item]) {
        let mut all_items = Vec::new();
        collect_all_items(&program.items, &mut all_items);
        for prog in &self.imported_programs {
            collect_all_items(&prog.items, &mut all_items);
        }

        let mut resources = Vec::new();
        for item in &all_items {
            if let Item::Struct(s) = item {
                if (s.is_resource || s.is_game) && (s.fields.is_empty() || s.fields.iter().all(|f| f.default.is_some())) {
                    resources.push(s.name.clone());
                }
            }
        }

        let mut startup_systems: Vec<ScheduledSystem> = Vec::new();
        let mut update_systems: Vec<ScheduledSystem> = Vec::new();
        let mut fixed_update_systems: Vec<ScheduledSystem> = Vec::new();
        let mut observers: Vec<String> = Vec::new();
        let mut has_camera_in_startup = false;

        let check_camera = |body: &[Stmt]| -> bool {
            let mut mentions = false;
            let mut on_expr = |e: &mut Expr| {
                match &e.kind {
                    ExprKind::Var(v) if v.eq_ignore_ascii_case("camera2d") || v.eq_ignore_ascii_case("camera3d") => {
                        mentions = true;
                    }
                    ExprKind::Call(callee, _) => {
                        if let ExprKind::Var(v) = &callee.kind {
                            if v.eq_ignore_ascii_case("camera2d") || v.eq_ignore_ascii_case("camera3d") {
                                mentions = true;
                            }
                        }
                    }
                    _ => {}
                }
            };
            let mut on_stmt = |_s: &mut Stmt| {};
            Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body.to_vec());
            mentions
        };

        let process_system = |
            attrs: &[Attr],
            fn_name: String,
            is_query: bool,
            is_obs_by_param: bool,
            body: &[Stmt],
            startup_systems: &mut Vec<ScheduledSystem>,
            update_systems: &mut Vec<ScheduledSystem>,
            fixed_update_systems: &mut Vec<ScheduledSystem>,
            observers: &mut Vec<String>,
            has_camera: &mut bool,
        | {
            let is_observer = is_obs_by_param || attrs.iter().any(|a| {
                a.name.eq_ignore_ascii_case("observer") || a.name.eq_ignore_ascii_case("observe")
            });

            let is_startup = attrs.iter().any(|a| {
                a.name.eq_ignore_ascii_case("startup")
                    || (a.name.eq_ignore_ascii_case("schedule") && a.args.iter().any(|x| x.eq_ignore_ascii_case("startup")))
            });

            let is_fixed = attrs.iter().any(|a| {
                a.name.eq_ignore_ascii_case("fixedupdate")
                    || a.name.eq_ignore_ascii_case("fixed_update")
                    || (a.name.eq_ignore_ascii_case("schedule") && a.args.iter().any(|x| x.eq_ignore_ascii_case("fixedupdate") || x.eq_ignore_ascii_case("fixed_update")))
            });

            let is_update = attrs.iter().any(|a| {
                a.name.eq_ignore_ascii_case("update")
                    || (a.name.eq_ignore_ascii_case("schedule") && a.args.iter().any(|x| x.eq_ignore_ascii_case("update")))
            }) || (is_query && !is_startup && !is_fixed && !is_observer);

            if !is_observer && !is_startup && !is_fixed && !is_update {
                return;
            }

            let is_chain = attrs.iter().any(|a| {
                a.name.eq_ignore_ascii_case("chain")
                    || (a.name.eq_ignore_ascii_case("schedule") && a.args.iter().any(|x| x.eq_ignore_ascii_case("chain")))
            });

            let after: Vec<String> = attrs.iter()
                .filter(|a| a.name.eq_ignore_ascii_case("after"))
                .flat_map(|a| a.args.clone())
                .map(|s| escape_rust_keyword(&camel_to_snake(&s)))
                .collect();

            let before: Vec<String> = attrs.iter()
                .filter(|a| a.name.eq_ignore_ascii_case("before"))
                .flat_map(|a| a.args.clone())
                .map(|s| escape_rust_keyword(&camel_to_snake(&s)))
                .collect();

            let mut rendered = fn_name.clone();
            for a in after {
                rendered = format!("{}.after({})", rendered, a);
            }
            for b in before {
                rendered = format!("{}.before({})", rendered, b);
            }

            if is_observer {
                observers.push(fn_name);
            } else if is_startup {
                if check_camera(body) {
                    *has_camera = true;
                }
                startup_systems.push(ScheduledSystem { rendered, chain: is_chain });
            } else if is_fixed {
                fixed_update_systems.push(ScheduledSystem { rendered, chain: is_chain });
            } else {
                update_systems.push(ScheduledSystem { rendered, chain: is_chain });
            }
        };

        for item in &all_items {
            match item {
                Item::Struct(s) => {
                    for m in &s.methods {
                        if self.is_ecs_fn(m) {
                            let fn_name = escape_rust_keyword(&format!("{}_{}", camel_to_snake(&s.name), camel_to_snake(&m.name)));
                            let is_obs_by_param = m.params.first().and_then(|p| p.ty.as_ref()).map(|ty| {
                                match ty {
                                    Type::Generic(g, _) => g == "On" || g == "Trigger",
                                    Type::Named(n) => n.starts_with("On<") || n.starts_with("Trigger<"),
                                    _ => false,
                                }
                            }).unwrap_or(false);
                            process_system(
                                &m.attrs,
                                fn_name,
                                true,
                                is_obs_by_param,
                                &m.body,
                                &mut startup_systems,
                                &mut update_systems,
                                &mut fixed_update_systems,
                                &mut observers,
                                &mut has_camera_in_startup,
                            );
                        }
                    }
                }
                Item::Fn(f) => {
                    let is_obs_by_param = f.params.first().and_then(|p| p.ty.as_ref()).map(|ty| {
                        match ty {
                            Type::Generic(g, _) => g == "On" || g == "Trigger",
                            Type::Named(n) => n.starts_with("On<") || n.starts_with("Trigger<"),
                            _ => false,
                        }
                    }).unwrap_or(false);

                    let is_ecs = self.is_ecs_fn(f);
                    let fn_name = escape_rust_keyword(&camel_to_snake(&f.name));
                    process_system(
                        &f.attrs,
                        fn_name,
                        is_ecs,
                        is_obs_by_param,
                        &f.body,
                        &mut startup_systems,
                        &mut update_systems,
                        &mut fixed_update_systems,
                        &mut observers,
                        &mut has_camera_in_startup,
                    );
                }
                _ => {}
            }
        }

        if !has_camera_in_startup {
            self.line("fn __boring_default_camera(mut commands: Commands) {");
            self.indent += 1;
            self.line("commands.spawn(Camera2d::default());");
            self.indent -= 1;
            self.line("}");
            self.blank();
            startup_systems.insert(0, ScheduledSystem {
                rendered: "__boring_default_camera".to_string(),
                chain: false,
            });
        }

        let mut title = game.name.clone();
        let mut width = 800.0f64;
        let mut height = 600.0f64;
        let mut clear_color_expr: Option<String> = None;

        for f in &game.fields {
            match f.name.as_str() {
                "title" => {
                    if let Some(Expr { kind: ExprKind::Str(s), .. }) = &f.default {
                        title = s.clone();
                    }
                }
                "width" => {
                    if let Some(Expr { kind: ExprKind::Int(w), .. }) = &f.default {
                        width = *w as f64;
                    } else if let Some(Expr { kind: ExprKind::Float(w), .. }) = &f.default {
                        width = *w;
                    }
                }
                "height" => {
                    if let Some(Expr { kind: ExprKind::Int(h), .. }) = &f.default {
                        height = *h as f64;
                    } else if let Some(Expr { kind: ExprKind::Float(h), .. }) = &f.default {
                        height = *h;
                    }
                }
                "clear_color" | "clearColor" | "background_color" | "backgroundColor" => {
                    if let Some(default_expr) = &f.default {
                        clear_color_expr = Some(self.emit_expr(default_expr));
                    }
                }
                _ => {}
            }
        }

        self.line("pub fn build_app() -> App {");
        self.indent += 1;
        self.line("let mut app = App::new();");
        self.line("app.add_plugins(DefaultPlugins.set(WindowPlugin {");
        self.indent += 1;
        self.line("primary_window: Some(Window {");
        self.indent += 1;
        self.line(&format!("title: \"{}\".into(),", escape_str(&title)));
        self.line(&format!("resolution: ({}u32, {}u32).into(),", width as u32, height as u32));
        self.line("..default()");
        self.indent -= 1;
        self.line("}),");
        self.line("..default()");
        self.indent -= 1;
        self.line("}));");

        if let Some(cc) = clear_color_expr {
            self.line(&format!("app.insert_resource(ClearColor({}));", cc));
        }

        for res in &resources {
            self.line(&format!("app.init_resource::<{}>();", res));
        }

        let emit_systems = |this: &mut Self, schedule: &str, systems: &[ScheduledSystem]| {
            if systems.is_empty() { return; }
            for chunk in systems.chunks(15) {
                let names: Vec<String> = chunk.iter().map(|s| s.rendered.clone()).collect();
                let any_chain = chunk.iter().any(|s| s.chain);
                if names.len() == 1 {
                    this.line(&format!("app.add_systems({}, {});", schedule, names[0]));
                } else if any_chain {
                    this.line(&format!("app.add_systems({}, ({}).chain());", schedule, names.join(", ")));
                } else {
                    this.line(&format!("app.add_systems({}, ({}));", schedule, names.join(", ")));
                }
            }
        };

        emit_systems(self, "Startup", &startup_systems);
        emit_systems(self, "Update", &update_systems);
        emit_systems(self, "FixedUpdate", &fixed_update_systems);

        for obs in &observers {
            self.line(&format!("app.add_observer({});", obs));
        }

        for s in extra_stmts {
            if matches!(s, Item::Stmt(Stmt::Comment(_))) {
                continue;
            }
            self.emit_item(s);
        }

        self.line("app");
        self.indent -= 1;
        self.line("}");
    }
}
