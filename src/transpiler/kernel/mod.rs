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

// Boring → Linux kernel module transpiler.
//
// Entry point: `transpile_kernel(program)` returns a `KernelTranspileOutput`.
// The generated code targets no_std + the `kernel` crate provided by the
// Linux build system (not tokio).

use crate::ast::{Program, Item, Stmt, ExprKind, Expr};

mod helpers;
mod emit_top;
mod emit_stmt;
mod emit_expr;

use helpers::KernelTranspiler;

// ─── Output type ─────────────────────────────────────────────────────────────

pub struct KernelTranspileOutput {
    /// The generated Rust source code.
    pub code: String,
}

// ─── Public entry point ───────────────────────────────────────────────────────

pub fn transpile_kernel(program: &Program) -> KernelTranspileOutput {
    let mut t = KernelTranspiler::new();
    t.emit_kernel_program(program);
    KernelTranspileOutput { code: t.out }
}

// ─── Program emission ─────────────────────────────────────────────────────────

/// Return true if the program contains at least one `task def` function.
fn program_has_task_fns(program: &Program) -> bool {
    program.items.iter().any(|item| match item {
        Item::Fn(f) => f.task,
        Item::Struct(s) => s.methods.iter().any(|m| m.task),
        Item::Enum(e)   => e.methods.iter().any(|m| m.task),
        _ => false,
    })
}

/// Return true if any statement (recursively) contains a `channel<T, N>` call.
fn stmts_have_channel(stmts: &[Stmt]) -> bool {
    stmts.iter().any(stmt_has_channel)
}

fn stmts_have_dyn_channel(stmts: &[Stmt]) -> bool {
    stmts.iter().any(stmt_has_dyn_channel)
}

fn stmt_has_channel(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::LetDestructure(s) => expr_is_static_channel(&s.value),
        Stmt::Let(s) => s.value.as_ref().map(expr_is_static_channel).unwrap_or(false),
        Stmt::Expr(e) => expr_is_static_channel(e),
        Stmt::If(s) => {
            s.branches.iter().any(|(_, b)| stmts_have_channel(b))
                || s.else_body.as_ref().map(|b| stmts_have_channel(b)).unwrap_or(false)
        }
        Stmt::While(s) => stmts_have_channel(&s.body),
        Stmt::For(s)   => stmts_have_channel(&s.body),
        Stmt::Loop(s)  => stmts_have_channel(&s.body),
        _ => false,
    }
}

fn stmt_has_dyn_channel(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::LetDestructure(s) => expr_is_dyn_channel(&s.value),
        Stmt::Let(s) => s.value.as_ref().map(expr_is_dyn_channel).unwrap_or(false),
        Stmt::Expr(e) => expr_is_dyn_channel(e),
        Stmt::If(s) => {
            s.branches.iter().any(|(_, b)| stmts_have_dyn_channel(b))
                || s.else_body.as_ref().map(|b| stmts_have_dyn_channel(b)).unwrap_or(false)
        }
        Stmt::While(s) => stmts_have_dyn_channel(&s.body),
        Stmt::For(s)   => stmts_have_dyn_channel(&s.body),
        Stmt::Loop(s)  => stmts_have_dyn_channel(&s.body),
        _ => false,
    }
}

/// `channel<T, N>` — const-generic capacity (stack buffer).
fn expr_is_static_channel(expr: &Expr) -> bool {
    matches!(&expr.kind,
        ExprKind::GenericCall(callee, tys, _)
        if matches!(&callee.kind, ExprKind::Var(n) if n == "channel") && tys.len() >= 2)
}

/// `channel<T>(cap)` or `channel(cap)` — runtime capacity (heap buffer).
fn expr_is_dyn_channel(expr: &Expr) -> bool {
    matches!(&expr.kind,
        ExprKind::GenericCall(callee, tys, args)
        if matches!(&callee.kind, ExprKind::Var(n) if n == "channel")
            && tys.len() < 2 && !args.is_empty())
    || matches!(&expr.kind,
        ExprKind::Call(callee, args)
        if matches!(&callee.kind, ExprKind::Var(n) if n == "channel") && !args.is_empty())
}

fn expr_is_named_call(expr: &Expr, name: &str) -> bool {
    matches!(&expr.kind,
        ExprKind::GenericCall(callee, _, _)
        if matches!(&callee.kind, ExprKind::Var(n) if n == name))
    || matches!(&expr.kind,
        ExprKind::Call(callee, _)
        if matches!(&callee.kind, ExprKind::Var(n) if n == name))
}

fn program_has_oneshot(program: &Program) -> bool {
    program.items.iter().any(|item| match item {
        Item::Fn(f) => stmts_have_named_call(&f.body, "oneshot"),
        Item::Struct(s) => s.methods.iter().any(|m| stmts_have_named_call(&m.body, "oneshot")),
        Item::Enum(e)   => e.methods.iter().any(|m| stmts_have_named_call(&m.body, "oneshot")),
        _ => false,
    })
}

fn program_has_watch(program: &Program) -> bool {
    program.items.iter().any(|item| match item {
        Item::Fn(f) => stmts_have_named_call(&f.body, "watch"),
        Item::Struct(s) => s.methods.iter().any(|m| stmts_have_named_call(&m.body, "watch")),
        Item::Enum(e)   => e.methods.iter().any(|m| stmts_have_named_call(&m.body, "watch")),
        _ => false,
    })
}

/// `broadcast<T, N>` — const-generic capacity (stack buffer per receiver).
fn expr_is_static_broadcast(expr: &Expr) -> bool {
    matches!(&expr.kind,
        ExprKind::GenericCall(callee, tys, _)
        if matches!(&callee.kind, ExprKind::Var(n) if n == "broadcast") && tys.len() >= 2)
}

/// `broadcast<T>(cap)` or `broadcast(cap)` — runtime capacity (heap buffer per receiver).
fn expr_is_dyn_broadcast(expr: &Expr) -> bool {
    matches!(&expr.kind,
        ExprKind::GenericCall(callee, tys, args)
        if matches!(&callee.kind, ExprKind::Var(n) if n == "broadcast")
            && tys.len() < 2 && !args.is_empty())
    || matches!(&expr.kind,
        ExprKind::Call(callee, args)
        if matches!(&callee.kind, ExprKind::Var(n) if n == "broadcast") && !args.is_empty())
}

fn stmts_have_static_broadcast(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| stmt_has_expr(s, expr_is_static_broadcast))
}

fn stmts_have_dyn_broadcast(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| stmt_has_expr(s, expr_is_dyn_broadcast))
}

fn stmt_has_expr(stmt: &Stmt, pred: fn(&Expr) -> bool) -> bool {
    match stmt {
        Stmt::LetDestructure(s) => pred(&s.value),
        Stmt::Let(s) => s.value.as_ref().map(&pred).unwrap_or(false),
        Stmt::Expr(e) => pred(e),
        Stmt::If(s) => {
            s.branches.iter().any(|(_, b)| b.iter().any(|st| stmt_has_expr(st, pred)))
                || s.else_body.as_ref().map(|b| b.iter().any(|st| stmt_has_expr(st, pred))).unwrap_or(false)
        }
        Stmt::While(s) => s.body.iter().any(|st| stmt_has_expr(st, pred)),
        Stmt::For(s)   => s.body.iter().any(|st| stmt_has_expr(st, pred)),
        Stmt::Loop(s)  => s.body.iter().any(|st| stmt_has_expr(st, pred)),
        _ => false,
    }
}

fn program_has_static_broadcast(program: &Program) -> bool {
    program.items.iter().any(|item| match item {
        Item::Fn(f) => stmts_have_static_broadcast(&f.body),
        Item::Struct(s) => s.methods.iter().any(|m| stmts_have_static_broadcast(&m.body)),
        Item::Enum(e)   => e.methods.iter().any(|m| stmts_have_static_broadcast(&m.body)),
        _ => false,
    })
}

fn program_has_dyn_broadcast(program: &Program) -> bool {
    program.items.iter().any(|item| match item {
        Item::Fn(f) => stmts_have_dyn_broadcast(&f.body),
        Item::Struct(s) => s.methods.iter().any(|m| stmts_have_dyn_broadcast(&m.body)),
        Item::Enum(e)   => e.methods.iter().any(|m| stmts_have_dyn_broadcast(&m.body)),
        _ => false,
    })
}


fn stmts_have_named_call(stmts: &[Stmt], name: &str) -> bool {
    stmts.iter().any(|s| stmt_has_named_call(s, name))
}

fn stmt_has_named_call(stmt: &Stmt, name: &str) -> bool {
    match stmt {
        Stmt::LetDestructure(s) => expr_is_named_call(&s.value, name),
        Stmt::Let(s) => s.value.as_ref().map(|v| expr_is_named_call(v, name)).unwrap_or(false),
        Stmt::Expr(e) => expr_is_named_call(e, name),
        Stmt::If(s) => {
            s.branches.iter().any(|(_, b)| stmts_have_named_call(b, name))
                || s.else_body.as_ref().map(|b| stmts_have_named_call(b, name)).unwrap_or(false)
        }
        Stmt::While(s) => stmts_have_named_call(&s.body, name),
        Stmt::For(s)   => stmts_have_named_call(&s.body, name),
        Stmt::Loop(s)  => stmts_have_named_call(&s.body, name),
        _ => false,
    }
}

/// Return true if the program contains any `channel<T, N>` expression (const-generic capacity).
fn program_has_channel(program: &Program) -> bool {
    program.items.iter().any(|item| match item {
        Item::Fn(f) => stmts_have_channel(&f.body),
        Item::Struct(s) => s.methods.iter().any(|m| stmts_have_channel(&m.body)),
        Item::Enum(e)   => e.methods.iter().any(|m| stmts_have_channel(&m.body)),
        _ => false,
    })
}

/// Return true if the program contains any `channel<T>(cap)` expression (runtime capacity).
fn program_has_dyn_channel(program: &Program) -> bool {
    program.items.iter().any(|item| match item {
        Item::Fn(f) => stmts_have_dyn_channel(&f.body),
        Item::Struct(s) => s.methods.iter().any(|m| stmts_have_dyn_channel(&m.body)),
        Item::Enum(e)   => e.methods.iter().any(|m| stmts_have_dyn_channel(&m.body)),
        _ => false,
    })
}

/// Return true if the program contains at least one async `stream def` (needs KernelChan prelude).
/// Purely sequential streams (no `wait`, no task calls) are excluded — they become iterators.
fn program_has_stream(program: &Program) -> bool {
    let empty = std::collections::HashSet::new();
    program.items.iter().any(|item| match item {
        Item::Fn(f) => f.stream && !crate::transpiler::helpers::body_is_sequential(&f.body, &empty),
        Item::Struct(s) => s.methods.iter().any(|m| m.stream && !crate::transpiler::helpers::body_is_sequential(&m.body, &empty)),
        Item::Enum(e)   => e.methods.iter().any(|m| m.stream && !crate::transpiler::helpers::body_is_sequential(&m.body, &empty)),
        _ => false,
    })
}

impl KernelTranspiler {
    fn emit_kernel_program(&mut self, program: &Program) {
        // Minimal no_std prelude for kernel modules.
        self.line("// Generated by boring --target kernel");
        self.line("#![no_std]");
        self.line("extern crate kernel;");
        self.blank();

        let has_tasks         = program_has_task_fns(program);
        let has_chan          = program_has_channel(program);
        let has_dyn_chan      = program_has_dyn_channel(program);
        let has_stream        = program_has_stream(program);
        let has_oneshot       = program_has_oneshot(program);
        let has_watch         = program_has_watch(program);
        let has_broadcast     = program_has_static_broadcast(program);
        let has_dyn_broadcast = program_has_dyn_broadcast(program);

        // Emit shared `use kernel::prelude::Arc` once if needed by any prelude.
        if has_tasks || has_chan || has_dyn_chan || has_stream || has_oneshot || has_watch
            || has_broadcast || has_dyn_broadcast
        {
            self.line("use kernel::prelude::Arc;");
            self.blank();
        }

        // Emit KernelFuture<T> prelude if any task defs are present.
        if has_tasks {
            self.emit_kernel_future_prelude();
        }

        // Emit KernelChan (stack) if any channel<T,N>, stream def, or broadcast<T,N> is used.
        if has_chan || has_stream || has_broadcast {
            self.has_channel = true;
            self.has_stream = has_stream;
            self.emit_kernel_chan_prelude();
        }

        // Emit DynKernelChan (heap) if any channel<T>(cap) or broadcast<T>(cap) is used.
        if has_dyn_chan || has_dyn_broadcast {
            self.emit_kernel_dyn_chan_prelude();
        }

        if has_oneshot {
            self.emit_kernel_oneshot_prelude();
        }

        if has_watch {
            self.emit_kernel_watch_prelude();
        }

        if has_broadcast {
            self.emit_kernel_broadcast_prelude();
        }

        if has_dyn_broadcast {
            self.emit_kernel_dyn_broadcast_prelude();
        }

        for item in &program.items {
            self.emit_item(item);
            self.blank();
        }
    }

    fn emit_kernel_chan_prelude(&mut self) {
        self.line("/// Ring-buffer channel generated by boring channel<T, N>.");
        self.line("pub struct KernelChan<T, const N: usize> {");
        self.indent += 1;
        self.line("buf:       [Option<T>; N],");
        self.line("read_idx:  usize,");
        self.line("write_idx: usize,");
        self.line("count:     usize,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("pub struct KernelSender<T, const N: usize> {");
        self.indent += 1;
        self.line("inner: Arc<kernel::sync::Mutex<KernelChan<T, N>>>,");
        self.line("not_empty: Arc<kernel::sync::CondVar>,");
        self.line("not_full:  Arc<kernel::sync::CondVar>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("pub struct KernelReceiver<T, const N: usize> {");
        self.indent += 1;
        self.line("inner: Arc<kernel::sync::Mutex<KernelChan<T, N>>>,");
        self.line("not_empty: Arc<kernel::sync::CondVar>,");
        self.line("not_full:  Arc<kernel::sync::CondVar>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("impl<T, const N: usize> KernelSender<T, N> {");
        self.indent += 1;
        self.line("pub fn send(&self, value: T) {");
        self.indent += 1;
        self.line("let mut guard = self.inner.lock();");
        self.line("self.not_full.wait_while(&mut guard, |c| c.count == N);");
        self.line("guard.buf[guard.write_idx] = Some(value);");
        self.line("guard.write_idx = (guard.write_idx + 1) % N;");
        self.line("guard.count += 1;");
        self.line("self.not_empty.notify_one();");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("impl<T, const N: usize> KernelReceiver<T, N> {");
        self.indent += 1;
        self.line("pub fn recv(&self) -> T {");
        self.indent += 1;
        self.line("let mut guard = self.inner.lock();");
        self.line("self.not_empty.wait_while(&mut guard, |c| c.count == 0);");
        self.line("let value = guard.buf[guard.read_idx].take().unwrap();");
        self.line("guard.read_idx = (guard.read_idx + 1) % N;");
        self.line("guard.count -= 1;");
        self.line("self.not_full.notify_one();");
        self.line("value");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("fn kernel_channel<T, const N: usize>() -> (KernelSender<T, N>, KernelReceiver<T, N>) {");
        self.indent += 1;
        self.line("let inner = Arc::new(kernel::sync::Mutex::new(KernelChan {");
        self.indent += 1;
        self.line("buf: [const { None }; N],");
        self.line("read_idx: 0,");
        self.line("write_idx: 0,");
        self.line("count: 0,");
        self.indent -= 1;
        self.line("}));");
        self.line("let not_empty = Arc::new(kernel::sync::CondVar::new());");
        self.line("let not_full  = Arc::new(kernel::sync::CondVar::new());");
        self.line("(");
        self.indent += 1;
        self.line("KernelSender  { inner: Arc::clone(&inner), not_empty: Arc::clone(&not_empty), not_full: Arc::clone(&not_full) },");
        self.line("KernelReceiver { inner, not_empty, not_full },");
        self.indent -= 1;
        self.line(")");
        self.indent -= 1;
        self.line("}");
        self.blank();
    }

    fn emit_kernel_dyn_chan_prelude(&mut self) {
        self.line("/// Heap ring-buffer channel generated by boring channel<T>(cap).");
        self.line("pub struct DynKernelChan<T> {");
        self.indent += 1;
        self.line("buf:       kernel::prelude::Vec<Option<T>>,");
        self.line("cap:       usize,");
        self.line("read_idx:  usize,");
        self.line("write_idx: usize,");
        self.line("count:     usize,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("pub struct DynKernelSender<T> {");
        self.indent += 1;
        self.line("inner:     Arc<kernel::sync::Mutex<DynKernelChan<T>>>,");
        self.line("not_empty: Arc<kernel::sync::CondVar>,");
        self.line("not_full:  Arc<kernel::sync::CondVar>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("pub struct DynKernelReceiver<T> {");
        self.indent += 1;
        self.line("inner:     Arc<kernel::sync::Mutex<DynKernelChan<T>>>,");
        self.line("not_empty: Arc<kernel::sync::CondVar>,");
        self.line("not_full:  Arc<kernel::sync::CondVar>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("impl<T> DynKernelSender<T> {");
        self.indent += 1;
        self.line("pub fn send(&self, value: T) {");
        self.indent += 1;
        self.line("let mut guard = self.inner.lock();");
        self.line("self.not_full.wait_while(&mut guard, |c| c.count == c.cap);");
        self.line("let idx = guard.write_idx;");
        self.line("guard.buf[idx] = Some(value);");
        self.line("guard.write_idx = (guard.write_idx + 1) % guard.cap;");
        self.line("guard.count += 1;");
        self.line("self.not_empty.notify_one();");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("impl<T> DynKernelReceiver<T> {");
        self.indent += 1;
        self.line("pub fn recv(&self) -> T {");
        self.indent += 1;
        self.line("let mut guard = self.inner.lock();");
        self.line("self.not_empty.wait_while(&mut guard, |c| c.count == 0);");
        self.line("let idx = guard.read_idx;");
        self.line("let value = guard.buf[idx].take().unwrap();");
        self.line("guard.read_idx = (guard.read_idx + 1) % guard.cap;");
        self.line("guard.count -= 1;");
        self.line("self.not_full.notify_one();");
        self.line("value");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("fn dyn_kernel_channel<T>(cap: usize) -> (DynKernelSender<T>, DynKernelReceiver<T>) {");
        self.indent += 1;
        self.line("let mut buf = kernel::prelude::Vec::with_capacity(cap);");
        self.line("for _ in 0..cap { buf.push(None); }");
        self.line("let inner = Arc::new(kernel::sync::Mutex::new(DynKernelChan {");
        self.indent += 1;
        self.line("buf,");
        self.line("cap,");
        self.line("read_idx: 0,");
        self.line("write_idx: 0,");
        self.line("count: 0,");
        self.indent -= 1;
        self.line("}));");
        self.line("let not_empty = Arc::new(kernel::sync::CondVar::new());");
        self.line("let not_full  = Arc::new(kernel::sync::CondVar::new());");
        self.line("(");
        self.indent += 1;
        self.line("DynKernelSender   { inner: Arc::clone(&inner), not_empty: Arc::clone(&not_empty), not_full: Arc::clone(&not_full) },");
        self.line("DynKernelReceiver { inner, not_empty, not_full },");
        self.indent -= 1;
        self.line(")");
        self.indent -= 1;
        self.line("}");
        self.blank();
    }

    fn emit_kernel_broadcast_prelude(&mut self) {
        // Each receiver gets its own ring-buffer slot (KernelChan) + condvars.
        // The sender holds an Arc<Mutex<Vec<KernelBcastSlot>>> and clones the value
        // into every slot on send().  subscribe() allocates a new slot.
        self.line("/// Per-receiver slot for KernelBroadcastSender.");
        self.line("struct KernelBcastSlot<T, const N: usize> {");
        self.indent += 1;
        self.line("inner:     Arc<kernel::sync::Mutex<KernelChan<T, N>>>,");
        self.line("not_empty: Arc<kernel::sync::CondVar>,");
        self.line("not_full:  Arc<kernel::sync::CondVar>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("pub struct KernelBroadcastSender<T, const N: usize> {");
        self.indent += 1;
        self.line("slots: Arc<kernel::sync::Mutex<kernel::prelude::Vec<KernelBcastSlot<T, N>>>>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("pub struct KernelBroadcastReceiver<T, const N: usize> {");
        self.indent += 1;
        self.line("inner:     Arc<kernel::sync::Mutex<KernelChan<T, N>>>,");
        self.line("not_empty: Arc<kernel::sync::CondVar>,");
        self.line("not_full:  Arc<kernel::sync::CondVar>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("impl<T: Clone, const N: usize> KernelBroadcastSender<T, N> {");
        self.indent += 1;
        self.line("pub fn send(&self, value: T) {");
        self.indent += 1;
        self.line("let slots = self.slots.lock();");
        self.line("for slot in slots.iter() {");
        self.indent += 1;
        self.line("let mut guard = slot.inner.lock();");
        self.line("slot.not_full.wait_while(&mut guard, |c| c.count == N);");
        self.line("guard.buf[guard.write_idx] = Some(value.clone());");
        self.line("guard.write_idx = (guard.write_idx + 1) % N;");
        self.line("guard.count += 1;");
        self.line("slot.not_empty.notify_one();");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line("pub fn subscribe(&self) -> KernelBroadcastReceiver<T, N> {");
        self.indent += 1;
        self.line("let inner = Arc::new(kernel::sync::Mutex::new(KernelChan {");
        self.indent += 1;
        self.line("buf: [const { None }; N],");
        self.line("read_idx: 0,");
        self.line("write_idx: 0,");
        self.line("count: 0,");
        self.indent -= 1;
        self.line("}));");
        self.line("let not_empty = Arc::new(kernel::sync::CondVar::new());");
        self.line("let not_full  = Arc::new(kernel::sync::CondVar::new());");
        self.line("let slot = KernelBcastSlot {");
        self.indent += 1;
        self.line("inner: Arc::clone(&inner),");
        self.line("not_empty: Arc::clone(&not_empty),");
        self.line("not_full: Arc::clone(&not_full),");
        self.indent -= 1;
        self.line("};");
        self.line("self.slots.lock().try_push(slot).expect(\"broadcast subscribe OOM\");");
        self.line("KernelBroadcastReceiver { inner, not_empty, not_full }");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("impl<T, const N: usize> KernelBroadcastReceiver<T, N> {");
        self.indent += 1;
        self.line("pub fn recv(&self) -> T {");
        self.indent += 1;
        self.line("let mut guard = self.inner.lock();");
        self.line("self.not_empty.wait_while(&mut guard, |c| c.count == 0);");
        self.line("let value = guard.buf[guard.read_idx].take().unwrap();");
        self.line("guard.read_idx = (guard.read_idx + 1) % N;");
        self.line("guard.count -= 1;");
        self.line("self.not_full.notify_one();");
        self.line("value");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("fn kernel_broadcast<T: Clone, const N: usize>() -> KernelBroadcastSender<T, N> {");
        self.indent += 1;
        self.line("KernelBroadcastSender { slots: Arc::new(kernel::sync::Mutex::new(kernel::prelude::Vec::new())) }");
        self.indent -= 1;
        self.line("}");
        self.blank();
    }

    fn emit_kernel_dyn_broadcast_prelude(&mut self) {
        self.line("/// Per-receiver slot for DynKernelBroadcastSender.");
        self.line("struct DynKernelBcastSlot<T> {");
        self.indent += 1;
        self.line("inner:     Arc<kernel::sync::Mutex<DynKernelChan<T>>>,");
        self.line("not_empty: Arc<kernel::sync::CondVar>,");
        self.line("not_full:  Arc<kernel::sync::CondVar>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("pub struct DynKernelBroadcastSender<T> {");
        self.indent += 1;
        self.line("slots: Arc<kernel::sync::Mutex<kernel::prelude::Vec<DynKernelBcastSlot<T>>>>,");
        self.line("cap:   usize,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("pub struct DynKernelBroadcastReceiver<T> {");
        self.indent += 1;
        self.line("inner:     Arc<kernel::sync::Mutex<DynKernelChan<T>>>,");
        self.line("not_empty: Arc<kernel::sync::CondVar>,");
        self.line("not_full:  Arc<kernel::sync::CondVar>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("impl<T: Clone> DynKernelBroadcastSender<T> {");
        self.indent += 1;
        self.line("pub fn send(&self, value: T) {");
        self.indent += 1;
        self.line("let slots = self.slots.lock();");
        self.line("for slot in slots.iter() {");
        self.indent += 1;
        self.line("let mut guard = slot.inner.lock();");
        self.line("slot.not_full.wait_while(&mut guard, |c| c.count == c.cap);");
        self.line("let idx = guard.write_idx;");
        self.line("guard.buf[idx] = Some(value.clone());");
        self.line("guard.write_idx = (guard.write_idx + 1) % guard.cap;");
        self.line("guard.count += 1;");
        self.line("slot.not_empty.notify_one();");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line("pub fn subscribe(&self) -> DynKernelBroadcastReceiver<T> {");
        self.indent += 1;
        self.line("let cap = self.cap;");
        self.line("let mut buf = kernel::prelude::Vec::with_capacity(cap);");
        self.line("for _ in 0..cap { buf.push(None); }");
        self.line("let inner = Arc::new(kernel::sync::Mutex::new(DynKernelChan {");
        self.indent += 1;
        self.line("buf, cap, read_idx: 0, write_idx: 0, count: 0,");
        self.indent -= 1;
        self.line("}));");
        self.line("let not_empty = Arc::new(kernel::sync::CondVar::new());");
        self.line("let not_full  = Arc::new(kernel::sync::CondVar::new());");
        self.line("let slot = DynKernelBcastSlot {");
        self.indent += 1;
        self.line("inner: Arc::clone(&inner),");
        self.line("not_empty: Arc::clone(&not_empty),");
        self.line("not_full: Arc::clone(&not_full),");
        self.indent -= 1;
        self.line("};");
        self.line("self.slots.lock().try_push(slot).expect(\"broadcast subscribe OOM\");");
        self.line("DynKernelBroadcastReceiver { inner, not_empty, not_full }");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("impl<T> DynKernelBroadcastReceiver<T> {");
        self.indent += 1;
        self.line("pub fn recv(&self) -> T {");
        self.indent += 1;
        self.line("let mut guard = self.inner.lock();");
        self.line("self.not_empty.wait_while(&mut guard, |c| c.count == 0);");
        self.line("let idx = guard.read_idx;");
        self.line("let value = guard.buf[idx].take().unwrap();");
        self.line("guard.read_idx = (guard.read_idx + 1) % guard.cap;");
        self.line("guard.count -= 1;");
        self.line("self.not_full.notify_one();");
        self.line("value");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("fn dyn_kernel_broadcast<T: Clone>(cap: usize) -> DynKernelBroadcastSender<T> {");
        self.indent += 1;
        self.line("DynKernelBroadcastSender {");
        self.indent += 1;
        self.line("slots: Arc::new(kernel::sync::Mutex::new(kernel::prelude::Vec::new())),");
        self.line("cap,");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
    }

    fn emit_kernel_oneshot_prelude(&mut self) {
        self.line("/// One-shot channel generated by boring oneshot<T>().");
        self.line("pub struct KernelOneshotInner<T> {");
        self.indent += 1;
        self.line("value: Option<T>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("pub struct KernelOneshotSender<T> {");
        self.indent += 1;
        self.line("inner: Arc<kernel::sync::Mutex<KernelOneshotInner<T>>>,");
        self.line("ready: Arc<kernel::sync::CondVar>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("pub struct KernelOneshotReceiver<T> {");
        self.indent += 1;
        self.line("inner: Arc<kernel::sync::Mutex<KernelOneshotInner<T>>>,");
        self.line("ready: Arc<kernel::sync::CondVar>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("impl<T> KernelOneshotSender<T> {");
        self.indent += 1;
        self.line("pub fn send(self, value: T) {");
        self.indent += 1;
        self.line("let mut guard = self.inner.lock();");
        self.line("guard.value = Some(value);");
        self.line("self.ready.notify_one();");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("impl<T> KernelOneshotReceiver<T> {");
        self.indent += 1;
        self.line("pub fn recv(self) -> T {");
        self.indent += 1;
        self.line("let mut guard = self.inner.lock();");
        self.line("self.ready.wait_while(&mut guard, |s| s.value.is_none());");
        self.line("guard.value.take().unwrap()");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("fn kernel_oneshot<T>() -> (KernelOneshotSender<T>, KernelOneshotReceiver<T>) {");
        self.indent += 1;
        self.line("let inner = Arc::new(kernel::sync::Mutex::new(KernelOneshotInner { value: None }));");
        self.line("let ready = Arc::new(kernel::sync::CondVar::new());");
        self.line("(");
        self.indent += 1;
        self.line("KernelOneshotSender  { inner: Arc::clone(&inner), ready: Arc::clone(&ready) },");
        self.line("KernelOneshotReceiver { inner, ready },");
        self.indent -= 1;
        self.line(")");
        self.indent -= 1;
        self.line("}");
        self.blank();
    }

    fn emit_kernel_watch_prelude(&mut self) {
        self.line("/// Watch channel generated by boring watch<T>(initial).");
        self.line("pub struct KernelWatchInner<T> {");
        self.indent += 1;
        self.line("value: T,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("pub struct KernelWatchSender<T> {");
        self.indent += 1;
        self.line("inner:   Arc<kernel::sync::Mutex<KernelWatchInner<T>>>,");
        self.line("changed: Arc<kernel::sync::CondVar>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("pub struct KernelWatchReceiver<T> {");
        self.indent += 1;
        self.line("inner:   Arc<kernel::sync::Mutex<KernelWatchInner<T>>>,");
        self.line("changed: Arc<kernel::sync::CondVar>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("impl<T: Clone> KernelWatchSender<T> {");
        self.indent += 1;
        self.line("pub fn send(&self, value: T) {");
        self.indent += 1;
        self.line("let mut guard = self.inner.lock();");
        self.line("guard.value = value;");
        self.line("self.changed.notify_all();");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("impl<T: Clone> KernelWatchReceiver<T> {");
        self.indent += 1;
        // recv(): wait for the next change, then return the new value.
        self.line("pub fn recv(&mut self) -> T {");
        self.indent += 1;
        self.line("let mut guard = self.inner.lock();");
        self.line("self.changed.wait(&mut guard);");
        self.line("guard.value.clone()");
        self.indent -= 1;
        self.line("}");
        // value: borrow current value without waiting.
        self.line("pub fn value(&self) -> T {");
        self.indent += 1;
        self.line("self.inner.lock().value.clone()");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("fn kernel_watch<T: Clone>(initial: T) -> (KernelWatchSender<T>, KernelWatchReceiver<T>) {");
        self.indent += 1;
        self.line("let inner = Arc::new(kernel::sync::Mutex::new(KernelWatchInner { value: initial }));");
        self.line("let changed = Arc::new(kernel::sync::CondVar::new());");
        self.line("(");
        self.indent += 1;
        self.line("KernelWatchSender  { inner: Arc::clone(&inner), changed: Arc::clone(&changed) },");
        self.line("KernelWatchReceiver { inner, changed },");
        self.indent -= 1;
        self.line(")");
        self.indent -= 1;
        self.line("}");
        self.blank();
    }

    fn emit_kernel_future_prelude(&mut self) {
        self.line("/// Blocking handle returned by `task def` functions.");
        self.line("pub struct KernelFuture<T> {");
        self.indent += 1;
        self.line("result: Arc<kernel::sync::Mutex<Option<Result<T, kernel::error::Error>>>>,");
        self.line("done_cond: Arc<kernel::sync::CondVar>,");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("impl<T> KernelFuture<T> {");
        self.indent += 1;
        self.line("pub fn done(&self) -> bool {");
        self.indent += 1;
        self.line("match self.result.try_lock() {");
        self.indent += 1;
        self.line("Ok(guard) => guard.is_some(),");
        self.line("Err(_) => false,");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line("pub fn wait(self) -> Result<T, kernel::error::Error> {");
        self.indent += 1;
        self.line("let mut guard = self.result.lock();");
        self.line("self.done_cond.wait_while(&mut guard, |r| r.is_none());");
        self.line("guard.take().unwrap()");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
    }
}
