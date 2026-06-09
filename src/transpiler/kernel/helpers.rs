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

// KernelTranspiler — state and output helpers.
//
// Distinct from the standard Transpiler; does not inherit from it.
// Targets no_std + kernel crate (provided by the Linux build system).

use crate::ast::{Type, OwnerQual};

pub(super) struct KernelTranspiler {
    pub(super) out: String,
    pub(super) indent: usize,
    /// True if at least one `task def` was emitted — triggers KernelFuture prelude.
    pub(super) has_task_fns: bool,
    /// True if any `channel<T, N>` expression is present — triggers KernelChan prelude.
    pub(super) has_channel: bool,
    /// True if at least one `stream def` is present — triggers KernelChan prelude (KernelSender/KernelReceiver).
    pub(super) has_stream: bool,
    /// True when emitting the body of a `stream def` — `yield` → `this.tx.send(...)`.
    pub(super) in_stream_body: bool,
}

impl KernelTranspiler {
    pub(super) fn new() -> Self {
        KernelTranspiler {
            out: String::new(),
            indent: 0,
            has_task_fns: false,
            has_channel: false,
            has_stream: false,
            in_stream_body: false,
        }
    }

    // ── Output helpers ────────────────────────────────────────────────────

    fn ind(&self) -> String {
        "    ".repeat(self.indent)
    }

    pub(super) fn line(&mut self, s: &str) {
        let ind = self.ind();
        self.out.push_str(&ind);
        self.out.push_str(s);
        self.out.push('\n');
    }

    pub(super) fn blank(&mut self) {
        self.out.push('\n');
    }

    #[allow(dead_code)]
    pub(super) fn push(&mut self, s: &str) {
        self.out.push_str(s);
    }

    // ── Type emission ─────────────────────────────────────────────────────

    /// Translate a Boring type to its Rust-for-Linux equivalent.
    pub(super) fn emit_type(&self, ty: &Type) -> String {
        match ty {
            Type::Int   => "i64".into(),
            Type::Uint  => "u64".into(),
            // float is forbidden in kernel code; emit a comment as a fallback
            Type::Float => "/* float forbidden */ f64".into(),
            Type::Str   => "kernel::str::CString".into(),
            Type::Bool  => "bool".into(),
            Type::Nil | Type::Void => "()".into(),
            Type::Never => "!".into(),

            Type::Named(n) => match n.as_str() {
                "int"    => "i64".into(),
                "uint"   => "u64".into(),
                "float"  => "/* float forbidden */ f64".into(),
                "bool"   => "bool".into(),
                "string" | "str" => "kernel::str::CString".into(),
                "void"   => "()".into(),
                other    => other.to_string(),
            },

            Type::Optional(inner)  => format!("Option<{}>", self.emit_type(inner)),
            Type::Array(inner)     => format!("kernel::prelude::Vec<{}>", self.emit_type(inner)),
            Type::Tuple(elems)     => format!(
                "({})",
                elems.iter().map(|t| self.emit_type(t)).collect::<Vec<_>>().join(", ")
            ),
            // {K: V} → RBTree<K, V>
            Type::Dict(k, v) => format!(
                "kernel::rbtree::RBTree<{}, {}>",
                self.emit_type(k),
                self.emit_type(v)
            ),
            // {T} → RBTree<T, ()>
            Type::Set(inner) => format!(
                "kernel::rbtree::RBTree<{}, ()>",
                self.emit_type(inner)
            ),

            Type::TypeParam(n) => n.clone(),

            Type::Generic(name, args) => {
                if name == "Box" {
                    // Box<T> in kernel = Box<T, kernel::alloc::KVmalloc>
                    let inner = args.first()
                        .map(|t| self.emit_type(t))
                        .unwrap_or_else(|| "()".into());
                    return format!("Box<{}, kernel::alloc::KVmalloc>", inner);
                }
                format!(
                    "{}<{}>",
                    name,
                    args.iter().map(|t| self.emit_type(t)).collect::<Vec<_>>().join(", ")
                )
            }

            Type::Qualified(inner, qual) => match qual {
                // T' → Box<T, kernel::alloc::KVmalloc>
                OwnerQual::Owned => {
                    format!("Box<{}, kernel::alloc::KVmalloc>", self.emit_type(inner))
                }
                // T'stack → T
                OwnerQual::Stack | OwnerQual::Copy => self.emit_type(inner),
                // T'auto / T'task → Arc<T>  (no Rc in kernel)
                OwnerQual::Auto | OwnerQual::Task => {
                    format!("Arc<{}>", self.emit_type(inner))
                }
                // T'actor → Arc<kernel::sync::Mutex<T>>
                OwnerQual::Actor => {
                    format!("Arc<kernel::sync::Mutex<{}>>", self.emit_type(inner))
                }
                // T'guard → Arc<kernel::sync::RwLock<T>>
                OwnerQual::Guard => {
                    format!("Arc<kernel::sync::RwLock<{}>>", self.emit_type(inner))
                }
                // T'weak → Weak<T>
                OwnerQual::Weak => {
                    format!("Weak<{}>", self.emit_type(inner))
                }
                // T& → &T  /  T'shared& → &T
                OwnerQual::Borrow | OwnerQual::BorrowTask => {
                    format!("&{}", self.emit_type(inner))
                }
                // var T& → &mut T
                OwnerQual::BorrowMut => {
                    format!("&mut {}", self.emit_type(inner))
                }
                // T'const → &kernel::str::CStr for strings, &'static T otherwise
                OwnerQual::Const => match inner.as_ref() {
                    Type::Str => "&kernel::str::CStr".into(),
                    Type::Named(n) if n == "string" || n == "str" => "&kernel::str::CStr".into(),
                    _ => format!("&'static {}", self.emit_type(inner)),
                },
                OwnerQual::Lifetime(lt) => {
                    format!("&'{} {}", lt, self.emit_type(inner))
                }
                _ => format!("&{}", self.emit_type(inner)),
            },

            // throws → Result<T, kernel::error::Error>
            Type::Fn(ret, params, throws, _task, req) => {
                let ps = params.iter()
                    .map(|t| self.emit_type(t))
                    .collect::<Vec<_>>()
                    .join(", ");
                let base = ret.as_ref()
                    .map(|r| self.emit_type(r))
                    .unwrap_or_else(|| "()".into());
                let r = if *throws {
                    format!("Result<{}, kernel::error::Error>", base)
                } else {
                    base
                };
                let trait_name = if *req { "Fn" } else { "FnMut" };
                format!("impl {}({}) -> {}", trait_name, ps, r)
            }

            Type::Dyn(inner)  => format!("dyn {}", self.emit_type(inner)),
            Type::Impl(inner) => format!("impl {}", self.emit_type(inner)),
            Type::SelfAssoc(name) => format!("Self::{}", name),
            Type::AssocOf(base, assoc) => {
                let base_name = match base.as_ref() {
                    Type::Named(n)      => n.clone(),
                    Type::Generic(n, _) => n.clone(),
                    _ => return self.emit_type(base),
                };
                format!("{}::{}", base_name, assoc)
            }
        }
    }
}
