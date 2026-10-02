//! Last-use analysis: which by-value reads of a local can be a Rust *move* instead of a `.clone()`.
//!
//! Boring's value semantics say passing a collection/struct by value never invalidates the
//! caller's variable, so the transpiler emits `.clone()` at every by-value read of such a local
//! (`Holder(xs)`, `Wrapper.List(xs)`, `return h`, ...). That clone is only *needed* if the variable
//! is read again afterwards. This pass finds, per function body, the `Var` occurrences that are
//! provably the last thing that can touch their variable, so the emitter may skip the clone there.
//!
//! The analysis is deliberately conservative — every doubt resolves to "clone" (the previous
//! behaviour), never to "move":
//!
//! * name-based, not scope-based: any later mention of the same name anywhere (a shadowing
//!   binding, another branch, a nested closure, ...) makes the earlier use a non-last use;
//! * an occurrence inside a loop whose body the binding is declared *outside* of is never last
//!   (the next iteration would read the moved value);
//! * an occurrence inside a closure / `task` / nested `fn` / `try` body / kernel block is never last
//!   (those are emitted as Rust closures or async blocks; moving out of a captured variable
//!   is a different, stricter rule), but still counts as a later mention for earlier uses;
//! * `defer` bodies run at function exit, so they are walked *after* everything else;
//! * an occurrence that shares its statement with another mention of the same name is never last
//!   (`f(&v, Holder(v))`, `v.merge(Holder(v))` would otherwise move `v` while it is still
//!   borrowed — rustc E0505);
//! * only names introduced by a plain `let`/`var`/`mut` of a *syntactically owned* value
//!   (declared `[T]`/`{K=V}`/`{T}`/`Named` type, a collection literal/comprehension, or a
//!   constructor call) are eligible, and only if no other binding form (parameter, `for`
//!   variable, pattern, destructure, closure parameter, comprehension variable, ...) uses the
//!   same name — a parameter or pattern-bound name may be a Rust reference, where a move would
//!   change the type rather than skip a copy.
//!
//! The matches below are exhaustive on purpose (no `_` arm): a new AST variant must be classified
//! here, otherwise a use hidden inside it would silently be missed.

use std::collections::{HashMap, HashSet};

use crate::ast::*;

/// `(name, line, col)` of a `Var` occurrence that is the last use of its variable.
pub(crate) type LastUseSites = HashSet<(String, usize, usize)>;

struct Use {
    name: String,
    line: usize,
    col: usize,
    loop_depth: usize,
    /// Inside a closure / task / nested fn / try body / kernel block.
    opaque: bool,
    /// Id of the statement the occurrence belongs to (see `Walker::cur`).
    stmt: usize,
}

struct Walker<'a> {
    uses: Vec<Use>,
    loop_depth: usize,
    opaque: usize,
    /// Nesting of expression contexts: statements reached through an expression
    /// (`Block`/`Do`/`If`/`Match`/... used as a value) keep the id of the enclosing statement.
    expr_nest: usize,
    next_stmt: usize,
    cur: usize,
    /// Loop depth at each eligible `let` of a name.
    decls: HashMap<String, Vec<usize>>,
    /// Names that must never be treated as movable.
    tainted: HashSet<String>,
    defers: Vec<&'a [Stmt]>,
}

pub(crate) fn last_use_sites(params: &[Param], body: &[Stmt]) -> LastUseSites {
    let mut w = Walker {
        uses: Vec::new(),
        loop_depth: 0,
        opaque: 0,
        expr_nest: 0,
        next_stmt: 0,
        cur: 0,
        decls: HashMap::new(),
        tainted: HashSet::new(),
        defers: Vec::new(),
    };
    for p in params {
        w.tainted.insert(p.name.clone());
        if let Some(d) = &p.default {
            w.expr(d);
        }
    }
    w.tainted.insert("self".to_string());
    w.stmts(body);
    // `defer` blocks run at function exit, after everything else: walk them last, opaque, so
    // they never become a "last use" themselves but do keep earlier uses from being one.
    let mut i = 0;
    while i < w.defers.len() {
        let d = w.defers[i];
        w.opaque += 1;
        w.stmts(d);
        w.opaque -= 1;
        i += 1;
    }
    w.finish()
}

impl<'a> Walker<'a> {
    fn finish(self) -> LastUseSites {
        let mut out = LastUseSites::new();
        // Occurrences per (name, stmt) and per exact key — duplicates (a synthesized/cloned
        // node sharing a source position) or a statement with several mentions are never last.
        let mut per_stmt: HashMap<(&str, usize), usize> = HashMap::new();
        let mut per_key: HashMap<(&str, usize, usize), usize> = HashMap::new();
        let mut last_idx: HashMap<&str, usize> = HashMap::new();
        for (i, u) in self.uses.iter().enumerate() {
            *per_stmt.entry((u.name.as_str(), u.stmt)).or_default() += 1;
            *per_key.entry((u.name.as_str(), u.line, u.col)).or_default() += 1;
            last_idx.insert(u.name.as_str(), i);
        }
        for (i, u) in self.uses.iter().enumerate() {
            if last_idx.get(u.name.as_str()) != Some(&i) { continue; }
            if u.opaque || u.line == 0 || self.tainted.contains(&u.name) { continue; }
            let Some(decl_depths) = self.decls.get(&u.name) else { continue };
            // Declared in the same or an outer loop level than the use — never inside a
            // loop the declaration sits outside of.
            if decl_depths.iter().any(|d| u.loop_depth > *d) { continue; }
            if per_stmt.get(&(u.name.as_str(), u.stmt)) != Some(&1) { continue; }
            if per_key.get(&(u.name.as_str(), u.line, u.col)) != Some(&1) { continue; }
            out.insert((u.name.clone(), u.line, u.col));
        }
        out
    }

    fn fresh(&mut self) -> usize {
        self.next_stmt += 1;
        self.next_stmt
    }

    /// Start a new statement-id for a compound statement's header expression group.
    fn header(&mut self) {
        if self.expr_nest == 0 {
            self.cur = self.fresh();
        }
    }

    fn taint(&mut self, name: &str) {
        self.tainted.insert(name.to_string());
    }

    fn taint_pattern(&mut self, p: &Pattern) {
        match p {
            Pattern::Bind(n) => self.taint(n),
            Pattern::Variant(_, subs) | Pattern::Tuple(subs) => for s in subs { self.taint_pattern(s) },
            Pattern::Some(inner) => self.taint_pattern(inner),
            Pattern::Wildcard | Pattern::Lit(_) | Pattern::None => {}
        }
    }

    fn looped<F: FnOnce(&mut Self)>(&mut self, f: F) {
        self.loop_depth += 1;
        f(self);
        self.loop_depth -= 1;
    }

    fn opaquely<F: FnOnce(&mut Self)>(&mut self, f: F) {
        self.opaque += 1;
        f(self);
        self.opaque -= 1;
    }

    fn nested<F: FnOnce(&mut Self)>(&mut self, f: F) {
        self.expr_nest += 1;
        f(self);
        self.expr_nest -= 1;
    }

    fn stmts(&mut self, stmts: &'a [Stmt]) {
        for s in stmts { self.stmt(s); }
    }

    fn owned_let_init(s: &LetStmt) -> bool {
        match (&s.ty, &s.value) {
            (_, None) => false,
            (Some(t), Some(_)) => matches!(
                t.without_mut(),
                Type::Array(_) | Type::Dict(..) | Type::Set(_) | Type::Named(_)
            ),
            (None, Some(v)) => match &v.kind {
                ExprKind::Array(_) | ExprKind::Dict(_) | ExprKind::Set(_)
                | ExprKind::ArrayFill { .. } | ExprKind::ArrayAlloc { .. }
                | ExprKind::ArrayComp { .. } | ExprKind::ArrayCompIter { .. } => true,
                ExprKind::Call(callee, _) => matches!(
                    &callee.kind,
                    ExprKind::Var(n) if n.chars().next().is_some_and(|c| c.is_uppercase())
                ),
                _ => false,
            },
        }
    }

    fn cond_clauses(&mut self, clauses: &'a [CondClause]) {
        for c in clauses {
            self.header();
            match c {
                CondClause::Let(name, e) => { self.expr(e); self.taint(name); }
                CondClause::LetPat(p, e) => { self.expr(e); self.taint_pattern(p); }
                CondClause::Expr(e) => self.expr(e),
            }
        }
    }

    fn stmt(&mut self, s: &'a Stmt) {
        if self.expr_nest == 0 {
            self.cur = self.fresh();
        }
        match s {
            Stmt::Let(l) => {
                if let Some(v) = &l.value { self.expr(v); }
                let eligible = Self::owned_let_init(l)
                    && !l.is_static && !l.is_lazy && !matches!(l.binding, BindingKind::Lazy);
                if eligible {
                    self.decls.entry(l.name.clone()).or_default().push(self.loop_depth);
                } else {
                    self.taint(&l.name);
                }
            }
            Stmt::LetDestructure(d) => {
                self.expr(&d.value);
                for b in &d.bindings { self.taint(&b.name); }
            }
            Stmt::Return(r) => { if let Some(e) = &r.value { self.expr(e); } }
            Stmt::Throw(t) => { if let Some(e) = &t.value { self.expr(e); } }
            Stmt::Break(_, e) => { if let Some(e) = e { self.expr(e); } }
            Stmt::Continue(_) | Stmt::Comment(_) | Stmt::Alias(_) => {}
            Stmt::Wait(e, _) | Stmt::Yield(e, _) | Stmt::Expr(e) => self.expr(e),
            Stmt::If(i) => self.if_stmt(i),
            Stmt::IfLet(i) => {
                self.cond_clauses(&i.clauses);
                self.stmts(&i.then_body);
                for b in &i.elif_branches {
                    self.cond_clauses(&b.clauses);
                    self.stmts(&b.body);
                }
                if let Some(e) = &i.else_body { self.stmts(e); }
            }
            Stmt::Match(m) => self.match_stmt(m),
            Stmt::While(w) => self.looped(|s| {
                s.header();
                s.expr(&w.condition);
                s.stmts(&w.body);
            }),
            Stmt::WhileLet(w) => {
                self.taint(&w.name);
                if let Some(p) = &w.pattern { self.taint_pattern(p); }
                self.looped(|s| {
                    s.header();
                    s.expr(&w.value);
                    s.stmts(&w.body);
                });
            }
            Stmt::DoWhile(d) => self.looped(|s| {
                s.stmts(&d.body);
                s.header();
                s.expr(&d.condition);
            }),
            Stmt::Loop(l) => self.looped(|s| s.stmts(&l.body)),
            Stmt::For(f) => {
                for v in &f.vars { self.taint(v); }
                self.header();
                self.expr(&f.iterable);
                self.looped(|s| s.stmts(&f.body));
            }
            Stmt::Guard(g) => {
                match &g.cond {
                    GuardCond::Expr(e) => { self.header(); self.expr(e); }
                    GuardCond::Clauses(cs) => self.cond_clauses(cs),
                }
                self.stmts(&g.else_body);
            }
            // `try`/`catch` is emitted as a Rust closure / labeled block with `?` plumbing.
            Stmt::Try(t) => self.opaquely(|s| {
                s.stmts(&t.body);
                for c in &t.catch_clauses { s.stmts(&c.body); }
            }),
            Stmt::Defer(body) => self.defers.push(body),
            // A nested fn is its own scope; treat its body like a closure (never last, but a mention).
            Stmt::Fn(f) => self.opaquely(|s| {
                for p in &f.params { s.taint(&p.name); }
                s.stmts(&f.body);
            }),
            Stmt::Struct(_) | Stmt::Enum(_) | Stmt::Mod(_) => {}
            Stmt::KernelBlock(k) => self.opaquely(|s| s.stmts(&k.body)),
            Stmt::With(w) => self.stmts(&w.body),
        }
    }

    fn if_stmt(&mut self, i: &'a IfStmt) {
        for (cond, body) in &i.branches {
            self.header();
            self.expr(cond);
            self.stmts(body);
        }
        if let Some(e) = &i.else_body { self.stmts(e); }
    }

    fn match_stmt(&mut self, m: &'a MatchStmt) {
        self.header();
        self.expr(&m.subject);
        for arm in &m.arms {
            for p in &arm.patterns { self.taint_pattern(p); }
            if let Some(g) = &arm.guard { self.header(); self.expr(g); }
            match &arm.body {
                MatchBody::Expr(e) => { self.header(); self.expr(e); }
                MatchBody::Block(b) => self.stmts(b),
            }
        }
    }

    fn args(&mut self, args: &'a [Arg]) {
        for a in args { self.expr(&a.value); }
    }

    fn expr(&mut self, e: &'a Expr) {
        match &e.kind {
            ExprKind::Var(name) => self.uses.push(Use {
                name: name.clone(),
                line: e.line,
                col: e.col,
                loop_depth: self.loop_depth,
                opaque: self.opaque > 0,
                stmt: self.cur,
            }),
            ExprKind::Int(_) | ExprKind::UInt64(_) | ExprKind::Float(_) | ExprKind::Str(_)
            | ExprKind::Bool(_) | ExprKind::Nil | ExprKind::Void | ExprKind::DotIdent(_) => {}
            ExprKind::StringInterp(segs) => {
                for seg in segs {
                    match seg {
                        StringSegment::Expr(x) | StringSegment::FormattedExpr(x, _) => self.expr(x),
                        StringSegment::Lit(_) => {}
                    }
                }
            }
            ExprKind::BinOp(_, l, r) => { self.expr(l); self.expr(r); }
            ExprKind::UnaryOp(_, x) => self.expr(x),
            // Right-hand side first: it is evaluated before the place is written.
            ExprKind::Assign(target, value) | ExprKind::QuestionAssign(target, value) => {
                self.expr(value);
                self.expr(target);
            }
            ExprKind::Field(x, _) | ExprKind::OptionalField(x, _) | ExprKind::Cast(x, _)
            | ExprKind::RelabelCast(x, _) => self.expr(x),
            ExprKind::Index(x, i) => { self.expr(x); self.expr(i); }
            ExprKind::LabeledIndex(x, args) => { self.expr(x); self.args(args); }
            ExprKind::Call(f, args) | ExprKind::GenericCall(f, _, args) => {
                self.expr(f);
                self.args(args);
            }
            ExprKind::MethodCall(x, _, args) | ExprKind::OptionalMethodCall(x, _, args)
            | ExprKind::Pipe(x, _, args) => {
                self.expr(x);
                self.args(args);
            }
            ExprKind::New { arena, ctor } => {
                if let Some(a) = arena { self.expr(a); }
                self.expr(ctor);
            }
            ExprKind::KernelLaunch { config, kernel } => self.opaquely(|s| {
                for c in [&config.block, &config.grid, &config.after].into_iter().flatten() {
                    s.expr(c);
                }
                s.expr(kernel);
            }),
            ExprKind::TryElse(x, d) | ExprKind::Else(x, d) => {
                // `try x else d` is emitted as a match/closure around a throws call.
                if matches!(&e.kind, ExprKind::TryElse(..)) {
                    self.opaquely(|s| { s.expr(x); s.expr(d); });
                } else {
                    self.expr(x);
                    self.expr(d);
                }
            }
            ExprKind::TryElseBlock(body, els) => {
                self.taint("error");
                self.opaquely(|s| { s.nested(|s| { s.stmts(body); s.stmts(els); }); });
            }
            ExprKind::Array(xs) | ExprKind::Tuple(xs) | ExprKind::Set(xs) | ExprKind::JoinAll(xs) => {
                for x in xs { self.expr(x); }
            }
            ExprKind::MacroCall { args, .. } => { for x in args { self.expr(x); } }
            ExprKind::ArrayFill { value, count } => {
                self.expr(count);
                self.looped(|s| s.expr(value));
            }
            ExprKind::ArrayAlloc { count } => self.expr(count),
            ExprKind::ArrayComp { expr, var, count } => {
                self.taint(var);
                self.expr(count);
                self.looped(|s| s.expr(expr));
            }
            ExprKind::ArrayCompIter { expr, var, iter } => {
                self.taint(var);
                self.expr(iter);
                self.looped(|s| s.expr(expr));
            }
            ExprKind::LabeledArrayComp { expr, clauses } => {
                for (v, count) in clauses {
                    self.taint(v);
                    self.expr(count);
                }
                self.looped(|s| s.expr(expr));
            }
            ExprKind::Dict(pairs) => {
                for (k, v) in pairs { self.expr(k); self.expr(v); }
            }
            ExprKind::Range { start, end, .. } => { self.expr(start); self.expr(end); }
            ExprKind::SliceRange { start, end, .. } => {
                if let Some(s) = start { self.expr(s); }
                if let Some(x) = end { self.expr(x); }
            }
            ExprKind::Closure(params, _, body, _, _) => {
                for p in params {
                    self.taint(&p.name);
                    if let Some(d) = &p.default { self.expr(d); }
                }
                self.opaquely(|s| match body {
                    ClosureBody::Expr(x) => s.nested(|s| s.expr(x)),
                    ClosureBody::Block(b) => s.nested(|s| s.stmts(b)),
                });
            }
            ExprKind::If(i) => self.nested(|s| {
                for (cond, body) in &i.branches {
                    s.expr(cond);
                    s.stmts(body);
                }
                if let Some(eb) = &i.else_body { s.stmts(eb); }
            }),
            ExprKind::Match(m) => self.nested(|s| {
                s.expr(&m.subject);
                for arm in &m.arms {
                    for p in &arm.patterns { s.taint_pattern(p); }
                    if let Some(g) = &arm.guard { s.expr(g); }
                    match &arm.body {
                        MatchBody::Expr(x) => s.expr(x),
                        MatchBody::Block(b) => s.stmts(b),
                    }
                }
            }),
            ExprKind::Block(b) | ExprKind::Do(b) => self.nested(|s| s.stmts(b)),
            ExprKind::Loop(l) => self.nested(|s| s.looped(|s| s.stmts(&l.body))),
            ExprKind::Task(x) => self.opaquely(|s| s.expr(x)),
            ExprKind::TaskWithTimeout(d, b) => self.opaquely(|s| { s.expr(d); s.expr(b); }),
            ExprKind::TrailingArrayBlock { callee, args, body, .. } => self.opaquely(|s| {
                s.expr(callee);
                s.args(args);
                s.nested(|s| s.stmts(body));
            }),
        }
    }
}
