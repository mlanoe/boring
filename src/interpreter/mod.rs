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

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::path::PathBuf;
use std::rc::Rc;

use crate::ast::*;

pub(crate) mod exec;
pub(crate) mod eval_expr;
pub(crate) mod call;
pub(crate) mod methods;
pub mod gpu_profile;


// ─── Errors ──────────────────────────────────────────────────────────────────

/// The interpreter's runtime-error payload -- shares its definition with the
/// checker's/transpiler's error (and the checker's warning) types; see
/// `crate::errors::SourceError`'s doc comment.
pub use crate::errors::SourceError as RuntimeError;

fn err(msg: impl Into<String>, line: usize) -> Signal {
    Signal::Error(RuntimeError::at_line(msg, line))
}

fn err_at(msg: impl Into<String>, line: usize, col: usize) -> Signal {
    Signal::Error(RuntimeError::at(msg, line, col))
}

fn err_span(msg: impl Into<String>, line: usize, col: usize, len: usize) -> Signal {
    Signal::Error(RuntimeError::new(msg, line, col, len))
}

/// Check whether two overload FnDecls conflict and exit with an error if they do.
/// A conflict exists when there is a call-arity N at which both can be invoked and
/// all N parameter types are compatible — most commonly triggered by default params:
///   def fn(int n, string s = "x"):   # also callable as fn(int)
///   def fn(int n):                    # CONFLICT at arity 1
fn check_overload_conflict_or_exit(a: &FnDecl, b: &FnDecl, _line: usize) {
    let a_min = a.params.iter().filter(|p| p.default.is_none()).count();
    let b_min = b.params.iter().filter(|p| p.default.is_none()).count();
    let a_max = a.params.len();
    let b_max = b.params.len();
    let lo = a_min.max(b_min);
    let hi = a_max.min(b_max);
    for n in lo..=hi {
        let conflict = a.params[..n].iter()
            .zip(b.params[..n].iter())
            .all(|(pa, pb)| match (&pa.ty, &pb.ty) {
                (Some(ta), Some(tb)) => types_match_for_overload(ta, tb),
                _ => true,
            });
        if conflict {
            let fmt_params = |params: &[crate::ast::Param]| {
                params.iter().map(|p| {
                    let ty = p.ty.as_ref().map(fmt_type).unwrap_or_else(|| "_".into());
                    if p.default.is_some() { format!("{}=default", ty) } else { ty }
                }).collect::<Vec<_>>().join(", ")
            };
            eprintln!(
                "error: ambiguous overload for '{}' — \
                 '{}({})' and '{}({})' both match a call with {} argument(s)",
                a.name, a.name, fmt_params(&a.params),
                b.name, fmt_params(&b.params), n
            );
            std::process::exit(1);
        }
    }
}

fn fmt_type(ty: &Type) -> String {
    match ty {
        Type::Int    => "int".into(),
        Type::Uint   => "uint".into(),
        Type::Uint8  => "uint8".into(),
        Type::Int8   => "int8".into(),
        Type::Int16  => "int16".into(),
        Type::Int32  => "int32".into(),
        Type::Int64  => "int64".into(),
        Type::Int128 => "int128".into(),
        Type::Uint16 => "uint16".into(),
        Type::Uint32 => "uint32".into(),
        Type::Uint64 => "uint64".into(),
        Type::Uint128 => "uint128".into(),
        Type::Float  => "float".into(),
        Type::Bool   => "bool".into(),
        Type::Str    => "string".into(),
        Type::Void   => "void".into(),
        Type::Named(n) => n.clone(),
        Type::Array(inner) => format!("[{}]", fmt_type(inner)),
        Type::Optional(inner) => format!("{}?", fmt_type(inner)),
        Type::Qualified(inner, _) => fmt_type(inner),
        _ => "?".into(),
    }
}

fn types_match_for_overload(a: &Type, b: &Type) -> bool {
    use Type::*;
    match (a, b) {
        (Int, Int) | (Uint, Uint) | (Uint8, Uint8) | (Float, Float) | (Bool, Bool) | (Str, Str) => true,
        (Int8, Int8) | (Int16, Int16) | (Int32, Int32) | (Int64, Int64) | (Int128, Int128) => true,
        (Uint16, Uint16) | (Uint32, Uint32) | (Uint64, Uint64) | (Uint128, Uint128) => true,
        (Named(x), Named(y)) => x == y,
        (Named(n), t) | (t, Named(n)) => match n.as_str() {
            "int"    => matches!(t, Int),
            "uint"   => matches!(t, Uint),
            "uint8"  => matches!(t, Uint8),
            "int8"   => matches!(t, Int8),
            "int16"  => matches!(t, Int16),
            "int32"  => matches!(t, Int32),
            "int64"  => matches!(t, Int64),
            "int128" => matches!(t, Int128),
            "uint16" => matches!(t, Uint16),
            "uint32" => matches!(t, Uint32),
            "uint64" => matches!(t, Uint64),
            "uint128" => matches!(t, Uint128),
            "float"  => matches!(t, Float),
            "bool"   => matches!(t, Bool),
            "string" => matches!(t, Str),
            _ => false,
        },
        (Array(_), Array(_)) | (Dict(..), Dict(..)) | (Set(_), Set(_)) => true,
        _ => false,
    }
}

/// Returns true when two FnDecl have the same parameter signature (same count + same types).
/// Used by ext block merging to distinguish "override same overload" from "add new overload".
fn params_same_signature(a: &FnDecl, b: &FnDecl) -> bool {
    if a.params.len() != b.params.len() { return false; }
    a.params.iter().zip(b.params.iter()).all(|(pa, pb)| {
        match (&pa.ty, &pb.ty) {
            (None, None) => true,
            (Some(ta), Some(tb)) => types_match_for_overload(ta, tb),
            _ => false,
        }
    })
}

// ─── "Did you mean?" helpers ─────────────────────────────────────────────────

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let n = a.len();
    let m = b.len();
    if n == 0 { return m; }
    if m == 0 { return n; }
    let mut row_prev: Vec<usize> = (0..=m).collect();
    let mut row_curr = vec![0usize; m + 1];
    for i in 1..=n {
        row_curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            row_curr[j] = (row_prev[j] + 1)
                .min(row_curr[j - 1] + 1)
                .min(row_prev[j - 1] + cost);
        }
        std::mem::swap(&mut row_prev, &mut row_curr);
    }
    row_prev[m]
}

fn closest_name<'a>(name: &str, candidates: &'a [String]) -> Option<&'a str> {
    let threshold = (name.len() / 3).max(2);
    candidates.iter()
        .filter(|c| c.len() >= 2)
        .map(|c| (levenshtein(name, c), c.as_str()))
        .filter(|(d, _)| *d <= threshold)
        .min_by_key(|(d, _)| *d)
        .map(|(_, s)| s)
}

// ─── Values ──────────────────────────────────────────────────────────────────

pub type EnvRef = Rc<RefCell<Env>>;

#[derive(Debug, Clone)]
pub struct ObjectInner {
    pub type_name: String,
    pub fields: Vec<(String, Value)>,
}

fn make_object(type_name: String, fields: Vec<(String, Value)>) -> Value {
    Value::Object(Rc::new(RefCell::new(ObjectInner { type_name, fields })))
}

/// Opaque handle into a collection — the internal representation is hidden from
/// user code.  Only `collection[index]`, `firstIndex()`, `nextIndex()` and
/// `removeAt()` are valid operations on an `Index` value.
///
/// * `Array(pos)`      — zero-based position; supports read **and** write.
/// * `DictKey(key)`    — the entry's key; supports read **and** write of the value.
/// * `Set(pos)`        — zero-based position; supports read and `removeAt` only.
///   Writing through a `Set` index is **forbidden** (it would break the
///   uniqueness invariant).
#[derive(Debug, Clone, PartialEq)]
pub enum IndexValue {
    Array(usize),
    DictKey(Box<Value>),
    Set(usize),
}

impl IndexValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            IndexValue::Array(_)  => "ArrayIndex",
            IndexValue::DictKey(_) => "DictIndex",
            IndexValue::Set(_)    => "SetIndex",
        }
    }
}

impl fmt::Display for IndexValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexValue::Array(p)   => write!(f, "<ArrayIndex {}>", p),
            IndexValue::DictKey(k) => write!(f, "<DictIndex {}>", k),
            IndexValue::Set(p)     => write!(f, "<SetIndex {}>", p),
        }
    }
}

#[derive(Clone)]
pub enum Value {
    /// Declared but not yet assigned: `let v` / `var v` without `= expr`.
    /// Any read of this value before assignment is a runtime error.
    Uninitialized,
    /// Value was moved out of this binding. Any subsequent read is a runtime error.
    Moved(String),
    Nil,
    /// Unit value returned by void functions — distinct from Nil.
    Void,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Uint8(u8),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    Int128(i128),
    Uint16(u16),
    Uint32(u32),
    Uint64(u64),
    Uint128(u128),
    Float(f64),
    Str(String),
    Array(Rc<Vec<Value>>),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Set(Vec<Value>),
    Object(Rc<RefCell<ObjectInner>>),
    EnumVariant {
        type_name: String,
        variant: String,
        fields: Vec<Value>,
    },
    EnumNamespace {
        name: String,
        variants: HashMap<String, Value>,
        methods: Vec<FnDecl>,
        setters: Vec<SetDecl>,
        conversions: Vec<AsDecl>,
        protocols: Vec<String>,
        /// The environment at enum definition time, used as the parent scope
        /// when calling enum methods — mirrors `Struct::captured`.
        captured: EnvRef,
    },
    Struct {
        decl: StructDecl,
        captured: EnvRef,
    },
    Future(Box<Value>),
    Fn {
        decl: FnDecl,
        captured: EnvRef,
    },
    OverloadedFn {
        name: String,
        variants: Vec<(FnDecl, EnvRef)>,
    },
    Closure {
        params: Vec<Param>,
        body: ClosureBody,
        captured: EnvRef,
    },
    NativeFn {
        name: String,
        func: fn(&[Value], usize) -> Result<Value, Signal>,
    },
    Range {
        start: i64,
        end: i64,
        inclusive: bool,
    },
    /// Internal-only: a labeled argument value produced by eval_args_labeled.
    /// Never escapes into user-visible values.
    Labeled {
        label: String,
        value: Box<Value>,
    },
    /// A Rust type imported via `use std.*: TypeName`.
    /// Callable as a constructor; maps to the closest boring value equivalent.
    /// Unknown types produce an opaque Object.
    RustType { name: String },
    /// Opaque collection index returned by `firstIndex()` / `nextIndex()`.
    /// Used as a subscript operand: `collection[index]`.
    Index(IndexValue),
    /// Synchronous channel simulation.  Sender and receiver share the same
    /// `buf`; sender pushes, receiver drains via `collect_iterable`.
    Channel {
        buf: Rc<RefCell<VecDeque<Value>>>,
        closed: Rc<RefCell<bool>>,
        is_sender: bool,
    },
    /// GPU kernel handle (simulation mode).
    /// The kernel has already run sequentially; `done()` returns true, `wait()` returns the value.
    KernelHandle {
        result: Box<Value>,
    },
    /// GPU kernel struct registered by `kernel Name:`.
    /// Stores the declaration and the environment at definition time.
    KernelStruct {
        decl: crate::ast::KernelDecl,
        captured: EnvRef,
    },
    /// GPU device handle, produced by `GPU(n)` (simulation mode).
    GpuDevice(usize),
    /// `Screen` built-in (simulation mode).
    /// Tracks the current virtual canvas size and key/resize state.
    Screen {
        width:   Rc<RefCell<u64>>,
        height:  Rc<RefCell<u64>>,
        title:   String,
        /// Frame counter — incremented each time `present()` is called.
        frame:   Rc<RefCell<u64>>,
        /// True for exactly one frame when the window was resized.
        resized: Rc<RefCell<bool>>,
        /// Keys "pressed" this frame (set by tests / simulation harness).
        keys:    Rc<RefCell<Vec<String>>>,
        /// Pixel buffer written by the last `present()` call (PPM output).
        pixels:  Rc<RefCell<Vec<u32>>>,
    },
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Nil, Value::Nil) => true,
            (Value::Void, Value::Void) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Uint(a), Value::Uint(b)) => a == b,
            (Value::Uint8(a), Value::Uint8(b)) => a == b,
            (Value::Int8(a), Value::Int8(b)) => a == b,
            (Value::Int16(a), Value::Int16(b)) => a == b,
            (Value::Int32(a), Value::Int32(b)) => a == b,
            (Value::Int64(a), Value::Int64(b)) => a == b,
            (Value::Int128(a), Value::Int128(b)) => a == b,
            (Value::Uint16(a), Value::Uint16(b)) => a == b,
            (Value::Uint32(a), Value::Uint32(b)) => a == b,
            (Value::Uint64(a), Value::Uint64(b)) => a == b,
            (Value::Uint128(a), Value::Uint128(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => a == b,
            (Value::Tuple(a), Value::Tuple(b)) => a == b,
            (Value::Future(a), Value::Future(b)) => a == b,
            (Value::Dict(a), Value::Dict(b)) => {
                if a.len() != b.len() { return false; }
                a.iter().all(|(k, v)| b.iter().any(|(k2, v2)| k == k2 && v == v2))
            }
            (Value::Set(a), Value::Set(b)) => {
                if a.len() != b.len() { return false; }
                a.iter().all(|x| b.contains(x))
            }
            (Value::Object(a), Value::Object(b)) => Rc::ptr_eq(a, b),
            (Value::EnumVariant { type_name: tn1, variant: v1, fields: f1 },
             Value::EnumVariant { type_name: tn2, variant: v2, fields: f2 }) => {
                tn1 == tn2 && v1 == v2 && f1 == f2
            }
            (Value::RustType { name: a }, Value::RustType { name: b }) => a == b,
            (Value::Index(a), Value::Index(b)) => a == b,
            (Value::Channel { buf: a, is_sender: s1, .. }, Value::Channel { buf: b, is_sender: s2, .. }) => {
                Rc::ptr_eq(a, b) && s1 == s2
            }
            (Value::OverloadedFn { name: a, .. }, Value::OverloadedFn { name: b, .. }) => a == b,
            _ => false,
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Uninitialized => write!(f, "Uninitialized"),
            Value::Moved(name) => write!(f, "Moved({:?})", name),
            Value::Nil => write!(f, "Nil"),
            Value::Void => write!(f, "Void"),
            Value::Bool(b) => write!(f, "Bool({:?})", b),
            Value::Int(n) => write!(f, "Int({:?})", n),
            Value::Uint(n) => write!(f, "Uint({:?})", n),
            Value::Uint8(n) => write!(f, "Uint8({:?})", n),
            Value::Int8(n) => write!(f, "Int8({:?})", n),
            Value::Int16(n) => write!(f, "Int16({:?})", n),
            Value::Int32(n) => write!(f, "Int32({:?})", n),
            Value::Int64(n) => write!(f, "Int64({:?})", n),
            Value::Int128(n) => write!(f, "Int128({:?})", n),
            Value::Uint16(n) => write!(f, "Uint16({:?})", n),
            Value::Uint32(n) => write!(f, "Uint32({:?})", n),
            Value::Uint64(n) => write!(f, "Uint64({:?})", n),
            Value::Uint128(n) => write!(f, "Uint128({:?})", n),
            Value::Float(n) => write!(f, "Float({:?})", n),
            Value::Str(s) => write!(f, "Str({:?})", s),
            Value::Array(v) => write!(f, "Array({:?})", v),
            Value::Tuple(v) => write!(f, "Tuple({:?})", v),
            Value::Dict(_v) => write!(f, "Dict(...)"),
            Value::Set(v) => write!(f, "Set({:?})", v),
            Value::Future(v) => write!(f, "Future({:?})", v),
            Value::Object(inner) => write!(f, "Object({})", inner.borrow().type_name),
            Value::EnumVariant { type_name, variant, .. } => write!(f, "EnumVariant({}::{})", type_name, variant),
            Value::EnumNamespace { name, .. } => write!(f, "EnumNamespace({})", name),
            Value::Struct { decl, .. } => write!(f, "Struct({})", decl.name),
            Value::Fn { decl, .. } => write!(f, "Fn({})", decl.name),
            Value::Closure { .. } => write!(f, "Closure"),
            Value::NativeFn { name, .. } => write!(f, "NativeFn({})", name),
            Value::Range { start, end, inclusive: _ } => write!(f, "Range({}..{})", start, end),
            Value::Labeled { label, value } => write!(f, "Labeled({}={:?})", label, value),
            Value::RustType { name } => write!(f, "RustType({})", name),
            Value::Index(idx) => write!(f, "Index({:?})", idx),
            Value::Channel { is_sender, .. } => write!(f, "Channel({})", if *is_sender { "sender" } else { "receiver" }),
            Value::OverloadedFn { name, .. } => write!(f, "OverloadedFn({})", name),
            Value::KernelHandle { result } => write!(f, "KernelHandle({:?})", result),
            Value::KernelStruct { decl, .. } => write!(f, "KernelStruct({})", decl.name),
            Value::GpuDevice(n) => write!(f, "GpuDevice({})", n),
            Value::Screen { title, .. } => write!(f, "Screen({:?})", title),
        }
    }
}

impl Value {
    /// Take ownership of an `Rc`-wrapped array's data, cloning only if another
    /// `Value::Array` still shares the same backing `Vec` (copy-on-write).
    /// In the common single-owner case (e.g. a loop-local array being pushed
    /// to repeatedly) this is O(1) instead of an O(n) deep clone.
    pub(crate) fn rc_vec_into_owned(rc: Rc<Vec<Value>>) -> Vec<Value> {
        Rc::try_unwrap(rc).unwrap_or_else(|rc| (*rc).clone())
    }

    pub fn type_name(&self) -> String {
        match self {
            Value::Uninitialized => "Uninitialized".into(),
            Value::Moved(_) => "moved".into(),
            Value::Nil => "Nil".into(),
            Value::Void => "Void".into(),
            Value::Bool(_) => "Bool".into(),
            Value::Int(_) => "Int".into(),
            Value::Uint(_) => "Uint".into(),
            Value::Uint8(_) => "Uint8".into(),
            Value::Int8(_) => "Int8".into(),
            Value::Int16(_) => "Int16".into(),
            Value::Int32(_) => "Int32".into(),
            Value::Int64(_) => "Int64".into(),
            Value::Int128(_) => "Int128".into(),
            Value::Uint16(_) => "Uint16".into(),
            Value::Uint32(_) => "Uint32".into(),
            Value::Uint64(_) => "Uint64".into(),
            Value::Uint128(_) => "Uint128".into(),
            Value::Float(_) => "Float".into(),
            Value::Str(_) => "String".into(),
            Value::Array(_) => "Array".into(),
            Value::Tuple(_) => "Tuple".into(),
            Value::Dict(_) => "Dict".into(),
            Value::Set(_) => "Set".into(),
            Value::Future(_) => "Future".into(),
            Value::Object(inner) => inner.borrow().type_name.clone(),
            Value::EnumVariant { type_name, .. } => type_name.clone(),
            Value::EnumNamespace { name, .. } => name.clone(),
            Value::Struct { decl, .. } => decl.name.clone(),
            Value::Fn { decl, .. } => decl.name.clone(),
            Value::Closure { .. } => "Closure".into(),
            Value::NativeFn { name, .. } => name.clone(),
            Value::Range { .. } => "Range".into(),
            Value::Labeled { .. } => "Labeled".into(),
            Value::RustType { name } => name.clone(),
            Value::Index(idx) => idx.type_name().into(),
            Value::Channel { is_sender, .. } => if *is_sender { "Sender".into() } else { "Receiver".into() },
            Value::OverloadedFn { name, .. } => name.clone(),
            Value::KernelHandle { .. } => "KernelHandle".into(),
            Value::KernelStruct { decl, .. } => decl.name.clone(),
            Value::GpuDevice(_) => "GpuDevice".into(),
            Value::Screen { .. } => "Screen".into(),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Uninitialized => write!(f, "<uninitialized>"),
            Value::Moved(name) => write!(f, "<moved:{}>", name),
            Value::Nil => write!(f, "nil"),
            Value::Void => write!(f, "void"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(n) => write!(f, "{}", n),
            Value::Uint(n) => write!(f, "{}", n),
            Value::Uint8(n) => write!(f, "{}", n),
            Value::Int8(n) => write!(f, "{}", n),
            Value::Int16(n) => write!(f, "{}", n),
            Value::Int32(n) => write!(f, "{}", n),
            Value::Int64(n) => write!(f, "{}", n),
            Value::Int128(n) => write!(f, "{}", n),
            Value::Uint16(n) => write!(f, "{}", n),
            Value::Uint32(n) => write!(f, "{}", n),
            Value::Uint64(n) => write!(f, "{}", n),
            Value::Uint128(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::Str(s) => write!(f, "{}", s),
            Value::Array(elems) => {
                write!(f, "[")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, "]")
            }
            Value::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, ")")
            }
            Value::Dict(pairs) => {
                write!(f, "{{")?;
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Set(elems) => {
                write!(f, "{{")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, "}}")
            }
            Value::Object(inner) => {
                let inner = inner.borrow();
                write!(f, "{}{{", inner.type_name)?;
                for (i, (k, v)) in inner.fields.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::EnumVariant { variant, fields, .. } => {
                if fields.is_empty() {
                    write!(f, "{}", variant)
                } else {
                    write!(f, "{}(", variant)?;
                    for (i, v) in fields.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", v)?;
                    }
                    write!(f, ")")
                }
            }
            Value::Future(_) => write!(f, "<future>"),
            Value::EnumNamespace { name, .. } => write!(f, "<enum {}>", name),
            Value::Struct { decl, .. } => write!(f, "<struct {}>", decl.name),
            Value::Fn { decl, .. } => write!(f, "<fn {}>", decl.name),
            Value::Closure { .. } => write!(f, "<closure>"),
            Value::NativeFn { name, .. } => write!(f, "<fn {}>", name),
            Value::RustType { name } => write!(f, "<type {}>", name),
            Value::Range { start, end, inclusive } => {
                if *inclusive {
                    write!(f, "{}..{}", start, end)
                } else {
                    write!(f, "{}..<{}", start, end)
                }
            }
            Value::Labeled { label, value } => write!(f, "{}={}", label, value),
            Value::Index(idx) => write!(f, "{}", idx),
            Value::Channel { is_sender, .. } => write!(f, "<{}>", if *is_sender { "sender" } else { "receiver" }),
            Value::OverloadedFn { name, .. } => write!(f, "<overloaded fn {}>", name),
            Value::KernelHandle { result } => write!(f, "<KernelHandle: {}>", result),
            Value::KernelStruct { decl, .. } => write!(f, "<kernel {}>", decl.name),
            Value::GpuDevice(n) => write!(f, "<GPU {}>", n),
            Value::Screen { title, width, height, .. } => {
                write!(f, "Screen({}x{}, {:?})", width.borrow(), height.borrow(), title)
            }
        }
    }
}

// ─── Signals ─────────────────────────────────────────────────────────────────

// Signal is used for interpreter control flow; boxing Value variants would add heap allocations in the hot loop
#[allow(clippy::result_large_err)]
#[derive(Debug, Clone)]
pub enum Signal {
    Return(Value),
    Break(Value),   // carries the value from `break expr` (Value::Void for plain `break`)
    Continue,
    Exception(Value),
    Error(RuntimeError),
    /// Produced by `yield expr` inside a stream function; caught by `call_stream_fn`.
    Yield(Value, usize),
}


type Eval = Result<Value, Signal>;

// ─── Environment ─────────────────────────────────────────────────────────────

pub struct Env {
    pub parent: Option<EnvRef>,
    pub vars: HashMap<String, Value>,
    pub mutable: HashSet<String>,
    /// Variables declared with an owned-element collection type ([T'], {T'}, {K:T'}).
    /// Pushing/inserting into these invalidates the source variable of the element.
    pub owned_collections: HashSet<String>,
    /// Variables explicitly declared with a task-safe qualifier ('rc, 'static, 'copy, or ').
    /// These bypass the task-capture safety check even if their runtime value is a collection.
    pub task_safe_vars: HashSet<String>,
    /// Variables whose declared type has the Owned qualifier (T').
    /// When captured by a task, these are moved in (source invalidated).
    pub owned_vars: HashSet<String>,
    /// Variables declared as `var T'shared` — reassignable but NOT method-mutable.
    /// Arc<T> has no interior mutability: def methods are forbidden, req methods are fine.
    pub shared_bindings: HashSet<String>,
    /// Variables with interior mutability: `T'actor`, `T'guard`, `T'task`, `T'actor'task`, `T'guard'task`.
    /// `def` methods are allowed even on `let` bindings because the Mutex/RwLock provides the lock.
    pub actor_bindings: HashSet<String>,
    /// Variables declared with `lazy` — uninitialized until first `?=` assignment.
    /// After the first assignment the name is removed from this set (init-once semantics).
    pub lazy_vars: HashSet<String>,
}

impl fmt::Debug for Env {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Env")
            .field("vars", &self.vars.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Env {
    pub fn new_global() -> EnvRef {
        let env = Rc::new(RefCell::new(Env {
            parent: None,
            vars: HashMap::new(),
            mutable: HashSet::new(),
            owned_collections: HashSet::new(),
            task_safe_vars: HashSet::new(),
            owned_vars: HashSet::new(),
            shared_bindings: HashSet::new(),
            actor_bindings: HashSet::new(),
            lazy_vars: HashSet::new(),
        }));
        register_stdlib(&env);
        env
    }

    pub fn child(parent: EnvRef) -> EnvRef {
        Rc::new(RefCell::new(Env {
            parent: Some(parent),
            vars: HashMap::new(),
            mutable: HashSet::new(),
            owned_collections: HashSet::new(),
            task_safe_vars: HashSet::new(),
            owned_vars: HashSet::new(),
            shared_bindings: HashSet::new(),
            actor_bindings: HashSet::new(),
            lazy_vars: HashSet::new(),
        }))
    }

    pub fn all_bindings(&self) -> Vec<(String, Value)> {
        self.vars.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    /// Move a variable's value out of its slot, leaving `Value::Nil` behind.
    /// Unlike `get`, this does not clone — used to obtain exclusive ownership
    /// of a value (e.g. an array) so a mutating method can rewrite it in
    /// place (copy-on-write) instead of always deep-cloning. The caller must
    /// restore the slot (via `force_set`) before the value is observable again.
    pub fn take(&mut self, name: &str) -> Option<Value> {
        if let Some(slot) = self.vars.get_mut(name) {
            Some(std::mem::replace(slot, Value::Nil))
        } else if let Some(ref parent) = self.parent {
            parent.borrow_mut().take(name)
        } else {
            None
        }
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        if let Some(v) = self.vars.get(name) {
            Some(v.clone())
        } else if let Some(ref parent) = self.parent {
            parent.borrow().get(name)
        } else {
            None
        }
    }

    /// Returns Err if the variable exists but is immutable, Ok(false) if not found, Ok(true) if set.
    #[allow(clippy::result_unit_err)]
    pub fn set(&mut self, name: &str, value: Value) -> Result<bool, ()> {
        if self.vars.contains_key(name) {
            // Lazy vars can be set once (they are in lazy_vars until first assignment).
            if self.lazy_vars.contains(name) {
                self.vars.insert(name.to_string(), value);
                self.lazy_vars.remove(name);
                return Ok(true);
            }
            if !self.mutable.contains(name) {
                return Err(());
            }
            self.vars.insert(name.to_string(), value);
            Ok(true)
        } else if let Some(ref parent) = self.parent {
            parent.borrow_mut().set(name, value)
        } else {
            Ok(false)
        }
    }

    /// Declare a `lazy` variable: stores Nil and marks it as pending initialization.
    pub fn define_lazy(&mut self, name: &str) {
        self.vars.insert(name.to_string(), Value::Nil);
        self.lazy_vars.insert(name.to_string());
        self.mutable.remove(name);
    }

    /// Returns true if the variable is a lazy binding pending its first `?=` assignment.
    pub fn is_lazy(&self, name: &str) -> bool {
        if self.vars.contains_key(name) {
            self.lazy_vars.contains(name)
        } else if let Some(ref parent) = self.parent {
            parent.borrow().is_lazy(name)
        } else {
            false
        }
    }

    pub fn define(&mut self, name: &str, value: Value) {
        self.vars.insert(name.to_string(), value);
        // Shadowing: if a previous `var` binding exists in this scope,
        // re-declaring with `let` must make it immutable.
        self.mutable.remove(name);
    }

    /// Update an existing variable bypassing the immutability check.
    /// Used to write back a mutated struct after a method call.
    pub fn force_set(&mut self, name: &str, value: Value) -> bool {
        if self.vars.contains_key(name) {
            self.vars.insert(name.to_string(), value);
            true
        } else if let Some(ref parent) = self.parent {
            parent.borrow_mut().force_set(name, value)
        } else {
            false
        }
    }

    pub fn define_mut(&mut self, name: &str, value: Value) {
        self.vars.insert(name.to_string(), value);
        self.mutable.insert(name.to_string());
    }

    /// `var T'shared` — reassignable but NOT method-mutable (Arc<T> has no interior mutability).
    pub fn define_shared_mut(&mut self, name: &str, value: Value) {
        self.vars.insert(name.to_string(), value);
        self.mutable.insert(name.to_string());
        self.shared_bindings.insert(name.to_string());
    }

    /// Collect all variable names visible in this scope and its parents.
    pub fn all_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.vars.keys().cloned().collect();
        if let Some(ref parent) = self.parent {
            names.extend(parent.borrow().all_names());
        }
        names
    }

    /// True if the binding has interior mutability (`T'actor`, `T'guard`, task variants).
    /// `def` methods are allowed even on `let` bindings.
    pub fn is_actor(&self, name: &str) -> bool {
        if self.vars.contains_key(name) {
            self.actor_bindings.contains(name)
        } else if let Some(ref parent) = self.parent {
            parent.borrow().is_actor(name)
        } else {
            false
        }
    }

    pub fn mark_actor(&mut self, name: &str) {
        self.actor_bindings.insert(name.to_string());
    }

    /// True if the binding was declared as `var T'shared` (reassignable, not method-mutable).
    pub fn is_shared(&self, name: &str) -> bool {
        if self.vars.contains_key(name) {
            self.shared_bindings.contains(name)
        } else if let Some(ref parent) = self.parent {
            parent.borrow().is_shared(name)
        } else {
            false
        }
    }

    /// Remove a variable after ownership transfer.
    pub fn invalidate(&mut self, name: &str) {
        if self.vars.remove(name).is_some() {
            self.mutable.remove(name);
            self.owned_collections.remove(name);
            self.task_safe_vars.remove(name);
            self.owned_vars.remove(name);
            self.shared_bindings.remove(name);
            self.actor_bindings.remove(name);
        } else if let Some(ref parent) = self.parent {
            parent.borrow_mut().invalidate(name);
        }
    }

    /// Mark a variable as moved: keep the name in scope so reads produce a "use of moved value"
    /// error instead of the less informative "undefined variable" error.
    pub fn set_moved(&mut self, name: &str) {
        if self.vars.contains_key(name) {
            self.vars.insert(name.to_string(), Value::Moved(name.to_string()));
            self.mutable.remove(name);
        } else if let Some(ref parent) = self.parent {
            parent.borrow_mut().set_moved(name);
        }
    }

    /// Returns true if `name` was declared with `var` (mutable), false if `let` (immutable).
    /// Returns true if not found (treat unknown as mutable for bare assignment contexts).
    pub fn is_mutable(&self, name: &str) -> bool {
        if self.vars.contains_key(name) {
            self.mutable.contains(name)
        } else if let Some(ref parent) = self.parent {
            parent.borrow().is_mutable(name)
        } else {
            true // not found — treat as mutable (bare assignment will define it)
        }
    }

    /// Mark a variable as having an Owned qualifier (T') — move semantics when captured by task.
    pub fn mark_owned_var(&mut self, name: &str) {
        if self.vars.contains_key(name) {
            self.owned_vars.insert(name.to_string());
        } else if let Some(ref parent) = self.parent {
            parent.borrow_mut().mark_owned_var(name);
        }
    }

    /// Check if a variable was declared with the Owned qualifier.
    pub fn is_owned_var(&self, name: &str) -> bool {
        if self.owned_vars.contains(name) {
            true
        } else if let Some(ref parent) = self.parent {
            parent.borrow().is_owned_var(name)
        } else {
            false
        }
    }

    /// Mark a variable as task-safe (declared with 'rc, 'static, or 'copy qualifier).
    pub fn mark_task_safe(&mut self, name: &str) {
        if self.vars.contains_key(name) {
            self.task_safe_vars.insert(name.to_string());
        } else if let Some(ref parent) = self.parent {
            parent.borrow_mut().mark_task_safe(name);
        }
    }

    /// Check if a variable was declared task-safe.
    pub fn is_task_safe_var(&self, name: &str) -> bool {
        if self.task_safe_vars.contains(name) {
            true
        } else if let Some(ref parent) = self.parent {
            parent.borrow().is_task_safe_var(name)
        } else {
            false
        }
    }

    /// Mark a variable as an owned-element collection.
    pub fn mark_owned_collection(&mut self, name: &str) {
        if self.vars.contains_key(name) {
            self.owned_collections.insert(name.to_string());
        } else if let Some(ref parent) = self.parent {
            parent.borrow_mut().mark_owned_collection(name);
        }
    }

    /// Check if a variable is a known owned-element collection.
    pub fn is_owned_collection(&self, name: &str) -> bool {
        if self.owned_collections.contains(name) {
            true
        } else if let Some(ref parent) = self.parent {
            parent.borrow().is_owned_collection(name)
        } else {
            false
        }
    }
}

// ─── Stdlib ──────────────────────────────────────────────────────────────────

/// Defines a global fixed-width integer conversion builtin (`int8()`, `uint16()`, …).
/// Mirrors the hand-written `uint8()` builtin, generalized via `TryFrom` (implemented
/// by std for every pair among the fixed-width integer primitives) so one macro body
/// covers both signed and unsigned target widths.
macro_rules! define_int_conv_builtin {
    ($e:expr, $name:literal, $variant:ident, $ty:ty) => {
        $e.define($name, Value::NativeFn {
            name: $name.into(),
            func: |args, line| {
                if args.len() != 1 {
                    return Err(err(concat!($name, "() takes 1 argument"), line));
                }
                match &args[0] {
                    Value::$variant(n) => Ok(Value::$variant(*n)),
                    Value::Int(n) => <$ty>::try_from(*n)
                        .map(Value::$variant)
                        .map_err(|_| err(format!("cannot convert Int {} to {} (out of range)", n, $name), line)),
                    Value::Uint(n) => <$ty>::try_from(*n)
                        .map(Value::$variant)
                        .map_err(|_| err(format!("cannot convert Uint {} to {} (out of range)", n, $name), line)),
                    Value::Float(f) => {
                        if *f < <$ty>::MIN as f64 || *f > <$ty>::MAX as f64 {
                            Err(err(format!("cannot convert Float to {} (out of range)", $name), line))
                        } else {
                            Ok(Value::$variant(*f as $ty))
                        }
                    }
                    Value::Str(s) => s.trim().parse::<$ty>()
                        .map(Value::$variant)
                        .map_err(|_| err(format!("cannot convert '{}' to {}", s, $name), line)),
                    Value::Bool(b) => Ok(Value::$variant(if *b { 1 as $ty } else { 0 as $ty })),
                    other => Err(err(format!("cannot convert {} to {}", other.type_name(), $name), line)),
                }
            },
        });
    };
}

fn register_io_and_diagnostics_builtins(e: &mut Env) {
    e.define("print", Value::NativeFn {
        name: "print".into(),
        func: |args, line| {
            // Positional: `print "{}", expr` — first arg is a format string with `{}`
            if args.len() >= 2
                && matches!(&args[0], Value::Str(_)) {
                    println!("{}", Interpreter::macro_format(args, line)?);
                    return Ok(Value::Nil);
                }
            let parts: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
            println!("{}", parts.join(" "));
            Ok(Value::Nil)
        },
    });

    e.define("write", Value::NativeFn {
        name: "write".into(),
        func: |args, line| {
            // Positional: `write "{}", expr` — first arg is a format string with `{}`
            if args.len() >= 2
                && matches!(&args[0], Value::Str(_)) {
                    print!("{}", Interpreter::macro_format(args, line)?);
                    return Ok(Value::Nil);
                }
            let parts: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
            print!("{}", parts.join(" "));
            Ok(Value::Nil)
        },
    });

    // Log-level builtins: print to stderr with a level prefix.
    // In transpiled code these map to `log::error!` / `log::warn!` / etc.
    fn log_format(prefix: &str, args: &[Value], line: usize) -> Result<Value, Signal> {
        let msg = if args.len() >= 2 {
            if matches!(&args[0], Value::Str(_)) {
                Interpreter::macro_format(args, line)?
            } else {
                args.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join(" ")
            }
        } else {
            args.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join(" ")
        };
        eprintln!("[{}] {}", prefix, msg);
        Ok(Value::Nil)
    }
    e.define("error", Value::NativeFn {
        name: "error".into(),
        func: |args, line| log_format("ERROR", args, line),
    });
    e.define("warn", Value::NativeFn {
        name: "warn".into(),
        func: |args, line| log_format("WARN", args, line),
    });
    e.define("info", Value::NativeFn {
        name: "info".into(),
        func: |args, line| log_format("INFO", args, line),
    });
    e.define("debug", Value::NativeFn {
        name: "debug".into(),
        func: |args, line| log_format("DEBUG", args, line),
    });
    e.define("trace", Value::NativeFn {
        name: "trace".into(),
        func: |args, line| log_format("TRACE", args, line),
    });

    e.define("len", Value::NativeFn {
        name: "len".into(),
        func: |args, line| {
            if args.len() != 1 {
                return Err(err("len() takes 1 argument", line));
            }
            match &args[0] {
                Value::Str(s) => Ok(Value::Int(s.chars().count() as i64)),
                Value::Array(a) => Ok(Value::Int(a.len() as i64)),
                Value::Dict(d) => Ok(Value::Int(d.len() as i64)),
                Value::Set(s) => Ok(Value::Int(s.len() as i64)),
                Value::Tuple(t) => Ok(Value::Int(t.len() as i64)),
                other => Err(err(format!("len() not supported for {}", other.type_name()), line)),
            }
        },
    });

    e.define("assert", Value::NativeFn {
        name: "assert".into(),
        func: |args, line| {
            if args.is_empty() {
                return Err(err("assert() takes at least 1 argument", line));
            }
            match &args[0] {
                Value::Bool(true) => Ok(Value::Nil),
                Value::Bool(false) => {
                    let msg = if args.len() > 1 {
                        format!("{}", args[1])
                    } else {
                        "assertion failed".to_string()
                    };
                    Err(err(msg, line))
                }
                other => Err(err(format!("assert() requires Bool, got {}", other.type_name()), line)),
            }
        },
    });

    e.define("assert_eq", Value::NativeFn {
        name: "assert_eq".into(),
        func: |args, line| {
            if args.len() < 2 {
                return Err(err("assert_eq() takes 2 arguments", line));
            }
            if args[0] == args[1] {
                Ok(Value::Void)
            } else {
                let msg = if args.len() > 2 {
                    format!("{}: expected {} == {}", args[2], args[0], args[1])
                } else {
                    format!("assertion failed: {} != {}", args[0], args[1])
                };
                Err(err(msg, line))
            }
        },
    });

    e.define("assert_neq", Value::NativeFn {
        name: "assert_neq".into(),
        func: |args, line| {
            if args.len() < 2 {
                return Err(err("assert_neq() takes 2 arguments", line));
            }
            if args[0] != args[1] {
                Ok(Value::Void)
            } else {
                let msg = if args.len() > 2 {
                    format!("{}: expected {} != {}", args[2], args[0], args[1])
                } else {
                    format!("assertion failed: both sides equal {}", args[0])
                };
                Err(err(msg, line))
            }
        },
    });

    // `panic(message)` — unconditional fatal error, not catchable by try/catch.
    e.define("panic", Value::NativeFn {
        name: "panic".into(),
        func: |args, line| {
            let msg = if args.is_empty() {
                "explicit panic".to_string()
            } else {
                format!("{}", args[0])
            };
            Err(err(format!("panic: {}", msg), line))
        },
    });
}

fn register_numeric_conversion_builtins(e: &mut Env) {
    e.define("int", Value::NativeFn {
        name: "int".into(),
        func: |args, line| {
            if args.len() != 1 {
                return Err(err("int() takes 1 argument", line));
            }
            match &args[0] {
                Value::Int(n) => Ok(Value::Int(*n)),
                Value::Uint(n) => Ok(Value::Int(*n as i64)),
                Value::Uint8(n) => Ok(Value::Int(*n as i64)),
                Value::Int8(n) => Ok(Value::Int(*n as i64)),
                Value::Int16(n) => Ok(Value::Int(*n as i64)),
                Value::Int32(n) => Ok(Value::Int(*n as i64)),
                Value::Int64(n) => Ok(Value::Int(*n)),
                Value::Int128(n) => Ok(Value::Int(*n as i64)),
                Value::Uint16(n) => Ok(Value::Int(*n as i64)),
                Value::Uint32(n) => Ok(Value::Int(*n as i64)),
                Value::Uint64(n) => Ok(Value::Int(*n as i64)),
                Value::Uint128(n) => Ok(Value::Int(*n as i64)),
                Value::Float(f) => Ok(Value::Int(*f as i64)),
                Value::Str(s) => s.trim().parse::<i64>()
                    .map(Value::Int)
                    .map_err(|_| err(format!("cannot convert '{}' to Int", s), line)),
                Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
                other => Err(err(format!("cannot convert {} to Int", other.type_name()), line)),
            }
        },
    });

    e.define("uint", Value::NativeFn {
        name: "uint".into(),
        func: |args, line| {
            if args.len() != 1 {
                return Err(err("uint() takes 1 argument", line));
            }
            match &args[0] {
                Value::Uint(n) => Ok(Value::Uint(*n)),
                Value::Uint8(n) => Ok(Value::Uint(*n as u64)),
                Value::Uint16(n) => Ok(Value::Uint(*n as u64)),
                Value::Uint32(n) => Ok(Value::Uint(*n as u64)),
                Value::Uint64(n) => Ok(Value::Uint(*n)),
                Value::Uint128(n) => Ok(Value::Uint(*n as u64)),
                Value::Int(n) => {
                    if *n < 0 { Err(err(format!("cannot convert negative Int {} to Uint", n), line)) }
                    else { Ok(Value::Uint(*n as u64)) }
                }
                Value::Int8(n) => {
                    if *n < 0 { Err(err(format!("cannot convert negative Int8 {} to Uint", n), line)) }
                    else { Ok(Value::Uint(*n as u64)) }
                }
                Value::Int16(n) => {
                    if *n < 0 { Err(err(format!("cannot convert negative Int16 {} to Uint", n), line)) }
                    else { Ok(Value::Uint(*n as u64)) }
                }
                Value::Int32(n) => {
                    if *n < 0 { Err(err(format!("cannot convert negative Int32 {} to Uint", n), line)) }
                    else { Ok(Value::Uint(*n as u64)) }
                }
                Value::Int64(n) => {
                    if *n < 0 { Err(err(format!("cannot convert negative Int64 {} to Uint", n), line)) }
                    else { Ok(Value::Uint(*n as u64)) }
                }
                Value::Int128(n) => {
                    if *n < 0 { Err(err(format!("cannot convert negative Int128 {} to Uint", n), line)) }
                    else { Ok(Value::Uint(*n as u64)) }
                }
                Value::Float(f) => {
                    if *f < 0.0 { Err(err("cannot convert negative Float to Uint".to_string(), line)) }
                    else { Ok(Value::Uint(*f as u64)) }
                }
                Value::Str(s) => s.trim().parse::<u64>()
                    .map(Value::Uint)
                    .map_err(|_| err(format!("cannot convert '{}' to Uint", s), line)),
                Value::Bool(b) => Ok(Value::Uint(if *b { 1 } else { 0 })),
                other => Err(err(format!("cannot convert {} to Uint", other.type_name()), line)),
            }
        },
    });

    e.define("uint8", Value::NativeFn {
        name: "uint8".into(),
        func: |args, line| {
            if args.len() != 1 {
                return Err(err("uint8() takes 1 argument", line));
            }
            match &args[0] {
                Value::Uint8(n) => Ok(Value::Uint8(*n)),
                Value::Int(n) => {
                    if *n < 0 || *n > 255 { Err(err(format!("cannot convert Int {} to Uint8 (out of 0..=255 range)", n), line)) }
                    else { Ok(Value::Uint8(*n as u8)) }
                }
                Value::Uint(n) => {
                    if *n > 255 { Err(err(format!("cannot convert Uint {} to Uint8 (out of 0..=255 range)", n), line)) }
                    else { Ok(Value::Uint8(*n as u8)) }
                }
                Value::Float(f) => {
                    if *f < 0.0 || *f > 255.0 { Err(err("cannot convert Float to Uint8 (out of 0..=255 range)".to_string(), line)) }
                    else { Ok(Value::Uint8(*f as u8)) }
                }
                Value::Str(s) => s.trim().parse::<u8>()
                    .map(Value::Uint8)
                    .map_err(|_| err(format!("cannot convert '{}' to Uint8", s), line)),
                Value::Bool(b) => Ok(Value::Uint8(if *b { 1 } else { 0 })),
                other => Err(err(format!("cannot convert {} to Uint8", other.type_name()), line)),
            }
        },
    });

    define_int_conv_builtin!(e, "int8", Int8, i8);
    define_int_conv_builtin!(e, "int16", Int16, i16);
    define_int_conv_builtin!(e, "int32", Int32, i32);
    define_int_conv_builtin!(e, "int64", Int64, i64);
    define_int_conv_builtin!(e, "int128", Int128, i128);
    define_int_conv_builtin!(e, "uint16", Uint16, u16);
    define_int_conv_builtin!(e, "uint32", Uint32, u32);
    define_int_conv_builtin!(e, "uint64", Uint64, u64);
    define_int_conv_builtin!(e, "uint128", Uint128, u128);

    e.define("float", Value::NativeFn {
        name: "float".into(),
        func: |args, line| {
            if args.len() != 1 {
                return Err(err("float() takes 1 argument", line));
            }
            match &args[0] {
                Value::Float(f) => Ok(Value::Float(*f)),
                Value::Int(n)   => Ok(Value::Float(*n as f64)),
                Value::Uint(n)  => Ok(Value::Float(*n as f64)),
                Value::Uint8(n) => Ok(Value::Float(*n as f64)),
                Value::Int8(n)  => Ok(Value::Float(*n as f64)),
                Value::Int16(n) => Ok(Value::Float(*n as f64)),
                Value::Int32(n) => Ok(Value::Float(*n as f64)),
                Value::Int64(n) => Ok(Value::Float(*n as f64)),
                Value::Int128(n) => Ok(Value::Float(*n as f64)),
                Value::Uint16(n) => Ok(Value::Float(*n as f64)),
                Value::Uint32(n) => Ok(Value::Float(*n as f64)),
                Value::Uint64(n) => Ok(Value::Float(*n as f64)),
                Value::Uint128(n) => Ok(Value::Float(*n as f64)),
                Value::Bool(b)  => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
                Value::Str(s) => s.trim().parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| err(format!("cannot convert '{}' to Float", s), line)),
                other => Err(err(format!("cannot convert {} to Float", other.type_name()), line)),
            }
        },
    });

}

fn register_string_and_math_builtins(e: &mut Env) {
    e.define("str", Value::NativeFn {
        name: "str".into(),
        func: |args, line| {
            if args.is_empty() {
                return Err(err("str() takes at least 1 argument", line));
            }
            // String first arg → formatting (like format())
            if matches!(&args[0], Value::Str(_)) {
                return Ok(Value::Str(Interpreter::macro_format(args, line)?));
            }
            // Single non-string arg → conversion
            if args.len() != 1 {
                return Err(err("str() takes 1 argument for conversion", line));
            }
            Ok(Value::Str(format!("{}", args[0])))
        },
    });

    e.define("min", Value::NativeFn {
        name: "min".into(),
        func: |args, line| {
            if args.len() == 1 {
                // min(array)
                match &args[0] {
                    Value::Array(arr) => Ok(arr.iter().min_by(|a, b| match (a, b) {
                        (Value::Int(x), Value::Int(y)) => x.cmp(y),
                        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                        _ => std::cmp::Ordering::Equal,
                    }).cloned().unwrap_or(Value::Nil)),
                    _ => Err(Signal::Error(RuntimeError { message: "min: expected array or two values".into(), line, col: 0, len: 0 })),
                }
            } else {
                // min(a, b)
                let a = args.first().cloned().unwrap_or(Value::Nil);
                let b = args.get(1).cloned().unwrap_or(Value::Nil);
                match (&a, &b) {
                    (Value::Int(x), Value::Int(y)) => Ok(if x <= y { a } else { b }),
                    (Value::Float(x), Value::Float(y)) => Ok(if x <= y { a } else { b }),
                    (Value::Int(x), Value::Float(y)) => Ok(if (*x as f64) <= *y { a } else { b }),
                    (Value::Float(x), Value::Int(y)) => Ok(if *x <= (*y as f64) { a } else { b }),
                    _ => Err(Signal::Error(RuntimeError { message: "min: expected numbers".into(), line, col: 0, len: 0 })),
                }
            }
        },
    });
    e.define("max", Value::NativeFn {
        name: "max".into(),
        func: |args, line| {
            if args.len() == 1 {
                match &args[0] {
                    Value::Array(arr) => Ok(arr.iter().max_by(|a, b| match (a, b) {
                        (Value::Int(x), Value::Int(y)) => x.cmp(y),
                        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                        _ => std::cmp::Ordering::Equal,
                    }).cloned().unwrap_or(Value::Nil)),
                    _ => Err(Signal::Error(RuntimeError { message: "max: expected array or two values".into(), line, col: 0, len: 0 })),
                }
            } else {
                let a = args.first().cloned().unwrap_or(Value::Nil);
                let b = args.get(1).cloned().unwrap_or(Value::Nil);
                match (&a, &b) {
                    (Value::Int(x), Value::Int(y)) => Ok(if x >= y { a } else { b }),
                    (Value::Float(x), Value::Float(y)) => Ok(if x >= y { a } else { b }),
                    (Value::Int(x), Value::Float(y)) => Ok(if (*x as f64) >= *y { a } else { b }),
                    (Value::Float(x), Value::Int(y)) => Ok(if *x >= (*y as f64) { a } else { b }),
                    _ => Err(Signal::Error(RuntimeError { message: "max: expected numbers".into(), line, col: 0, len: 0 })),
                }
            }
        },
    });
    e.define("abs", Value::NativeFn {
        name: "abs".into(),
        func: |args, line| match args.first() {
            Some(Value::Int(n)) => Ok(Value::Int(n.abs())),
            Some(Value::Float(f)) => Ok(Value::Float(f.abs())),
            _ => Err(Signal::Error(RuntimeError { message: "abs: expected number".into(), line, col: 0, len: 0 })),
        },
    });
    e.define("floor", Value::NativeFn {
        name: "floor".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Int(f.floor() as i64)),
            Some(Value::Int(n)) => Ok(Value::Int(*n)),
            _ => Err(Signal::Error(RuntimeError { message: "floor: expected number".into(), line, col: 0, len: 0 })),
        },
    });
    e.define("ceil", Value::NativeFn {
        name: "ceil".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Int(f.ceil() as i64)),
            Some(Value::Int(n)) => Ok(Value::Int(*n)),
            _ => Err(Signal::Error(RuntimeError { message: "ceil: expected number".into(), line, col: 0, len: 0 })),
        },
    });
    e.define("round", Value::NativeFn {
        name: "round".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Int(f.round() as i64)),
            Some(Value::Int(n)) => Ok(Value::Int(*n)),
            _ => Err(Signal::Error(RuntimeError { message: "round: expected number".into(), line, col: 0, len: 0 })),
        },
    });
    e.define("sqrt", Value::NativeFn {
        name: "sqrt".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.sqrt())),
            Some(Value::Int(n)) => Ok(Value::Float((*n as f64).sqrt())),
            _ => Err(Signal::Error(RuntimeError { message: "sqrt: expected number".into(), line, col: 0, len: 0 })),
        },
    });
    e.define("readLine", Value::NativeFn {
        name: "readLine".into(),
        func: |_args, _line| {
            let mut buf = String::new();
            let n = std::io::stdin().read_line(&mut buf).unwrap_or(0);
            if n == 0 {
                Ok(Value::Nil)
            } else {
                Ok(Value::Str(buf.trim_end_matches('\n').to_string()))
            }
        },
    });

    // ─── Math functions ───────────────────────────────────────────────────────

    e.define("pow", Value::NativeFn {
        name: "pow".into(),
        func: |args, line| {
            let base = match args.first() {
                Some(Value::Float(f)) => *f,
                Some(Value::Int(n))   => *n as f64,
                _ => return Err(err("pow: expected numeric base", line)),
            };
            let exp = match args.get(1) {
                Some(Value::Float(f)) => *f,
                Some(Value::Int(n))   => *n as f64,
                _ => return Err(err("pow: expected numeric exponent", line)),
            };
            Ok(Value::Float(base.powf(exp)))
        },
    });
    e.define("log", Value::NativeFn {
        name: "log".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.ln())),
            Some(Value::Int(n))   => Ok(Value::Float((*n as f64).ln())),
            _ => Err(err("log: expected number", line)),
        },
    });
    e.define("log2", Value::NativeFn {
        name: "log2".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.log2())),
            Some(Value::Int(n))   => Ok(Value::Float((*n as f64).log2())),
            _ => Err(err("log2: expected number", line)),
        },
    });
    e.define("log10", Value::NativeFn {
        name: "log10".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.log10())),
            Some(Value::Int(n))   => Ok(Value::Float((*n as f64).log10())),
            _ => Err(err("log10: expected number", line)),
        },
    });
    e.define("sin", Value::NativeFn {
        name: "sin".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.sin())),
            Some(Value::Int(n))   => Ok(Value::Float((*n as f64).sin())),
            _ => Err(err("sin: expected number", line)),
        },
    });
    e.define("cos", Value::NativeFn {
        name: "cos".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.cos())),
            Some(Value::Int(n))   => Ok(Value::Float((*n as f64).cos())),
            _ => Err(err("cos: expected number", line)),
        },
    });
    e.define("tan", Value::NativeFn {
        name: "tan".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.tan())),
            Some(Value::Int(n))   => Ok(Value::Float((*n as f64).tan())),
            _ => Err(err("tan: expected number", line)),
        },
    });
    e.define("asin", Value::NativeFn {
        name: "asin".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.asin())),
            Some(Value::Int(n))   => Ok(Value::Float((*n as f64).asin())),
            _ => Err(err("asin: expected number", line)),
        },
    });
    e.define("acos", Value::NativeFn {
        name: "acos".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.acos())),
            Some(Value::Int(n))   => Ok(Value::Float((*n as f64).acos())),
            _ => Err(err("acos: expected number", line)),
        },
    });
    e.define("atan", Value::NativeFn {
        name: "atan".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.atan())),
            Some(Value::Int(n))   => Ok(Value::Float((*n as f64).atan())),
            _ => Err(err("atan: expected number", line)),
        },
    });
    e.define("atan2", Value::NativeFn {
        name: "atan2".into(),
        func: |args, line| {
            let y = match args.first() {
                Some(Value::Float(f)) => *f,
                Some(Value::Int(n))   => *n as f64,
                _ => return Err(err("atan2: expected numeric y", line)),
            };
            let x = match args.get(1) {
                Some(Value::Float(f)) => *f,
                Some(Value::Int(n))   => *n as f64,
                _ => return Err(err("atan2: expected numeric x", line)),
            };
            Ok(Value::Float(y.atan2(x)))
        },
    });
    e.define("exp", Value::NativeFn {
        name: "exp".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.exp())),
            Some(Value::Int(n))   => Ok(Value::Float((*n as f64).exp())),
            _ => Err(err("exp: expected number", line)),
        },
    });
    e.define("tanh", Value::NativeFn {
        name: "tanh".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.tanh())),
            Some(Value::Int(n))   => Ok(Value::Float((*n as f64).tanh())),
            _ => Err(err("tanh: expected number", line)),
        },
    });
    e.define("sum", Value::NativeFn {
        name: "sum".into(),
        func: |args, line| match args.first() {
            Some(Value::Array(arr)) => {
                let mut int_sum = 0i64;
                let mut float_sum = 0.0f64;
                let mut is_float = false;
                for v in arr.iter() {
                    match v {
                        Value::Float(f) => { float_sum += f; is_float = true; }
                        Value::Int(n)   => { int_sum += n; float_sum += *n as f64; }
                        _ => return Err(err("sum: array must contain numbers", line)),
                    }
                }
                if is_float { Ok(Value::Float(float_sum)) } else { Ok(Value::Int(int_sum)) }
            }
            _ => Err(err("sum: expected array", line)),
        },
    });
    e.define("bitsToFloat", Value::NativeFn {
        name: "bitsToFloat".into(),
        func: |args, line| match args.first() {
            Some(Value::Int(n)) => Ok(Value::Float(f32::from_bits(*n as u32) as f64)),
            _ => Err(err("bitsToFloat: expected int", line)),
        },
    });
    e.define("floatToBits", Value::NativeFn {
        name: "floatToBits".into(),
        func: |args, line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Int((*f as f32).to_bits() as i64)),
            Some(Value::Int(n))   => Ok(Value::Int((*n as f32).to_bits() as i64)),
            _ => Err(err("floatToBits: expected number", line)),
        },
    });
    e.define("chr", Value::NativeFn {
        name: "chr".into(),
        func: |args, line| match args.first() {
            Some(Value::Int(n)) => {
                char::from_u32(*n as u32)
                    .map(|c| Value::Str(c.to_string()))
                    .ok_or_else(|| err("chr: invalid Unicode code point", line))
            }
            _ => Err(err("chr: expected int", line)),
        },
    });
    e.define("ord", Value::NativeFn {
        name: "ord".into(),
        func: |args, line| match args.first() {
            Some(Value::Str(s)) => {
                s.chars().next()
                    .map(|c| Value::Int(c as i64))
                    .ok_or_else(|| err("ord: empty string", line))
            }
            _ => Err(err("ord: expected string", line)),
        },
    });
    e.define("clamp", Value::NativeFn {
        name: "clamp".into(),
        func: |args, line| {
            match (args.first(), args.get(1), args.get(2)) {
                (Some(Value::Float(x)), Some(Value::Float(lo)), Some(Value::Float(hi))) =>
                    Ok(Value::Float(x.clamp(*lo, *hi))),
                (Some(Value::Int(x)), Some(Value::Int(lo)), Some(Value::Int(hi))) =>
                    Ok(Value::Int(*x.clamp(lo, hi))),
                (Some(Value::Float(x)), Some(Value::Int(lo)), Some(Value::Int(hi))) =>
                    Ok(Value::Float(x.clamp(*lo as f64, *hi as f64))),
                (Some(Value::Int(x)), Some(Value::Float(lo)), Some(Value::Float(hi))) =>
                    Ok(Value::Float((*x as f64).clamp(*lo, *hi))),
                _ => Err(err("clamp: expected (number, min, max)", line)),
            }
        },
    });
    e.define("sign", Value::NativeFn {
        name: "sign".into(),
        func: |args, line| match args.first() {
            Some(Value::Int(n))   => Ok(Value::Int(n.signum())),
            Some(Value::Float(f)) => Ok(Value::Float(f.signum())),
            _ => Err(err("sign: expected number", line)),
        },
    });
    e.define("isNaN", Value::NativeFn {
        name: "isNaN".into(),
        func: |args, _line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Bool(f.is_nan())),
            Some(Value::Int(_))   => Ok(Value::Bool(false)),
            _ => Ok(Value::Bool(false)),
        },
    });
    e.define("isInfinite", Value::NativeFn {
        name: "isInfinite".into(),
        func: |args, _line| match args.first() {
            Some(Value::Float(f)) => Ok(Value::Bool(f.is_infinite())),
            _                     => Ok(Value::Bool(false)),
        },
    });

}

fn register_concurrency_builtins(e: &mut Env) {
    // ─── Math constants ───────────────────────────────────────────────────────

    e.define("drop", Value::NativeFn {
        name: "drop".into(),
        func: |_args, _line| {
            // In the interpreter, drop is a no-op: values are reference-counted anyway.
            Ok(Value::Nil)
        },
    });

    // wait(Duration) — async pause; no-op in the synchronous interpreter.
    // In transpiled Rust: tokio::time::sleep(dur).await
    // Signature: task wait(Duration duration) throws Error.Cancelled
    e.define("wait", Value::NativeFn {
        name: "wait".into(),
        func: |_args, _line| {
            // Interpreter has no async runtime — treat as instant return.
            Ok(Value::Nil)
        },
    });

    // channel(cap) — type-erased form; returns the same (Sender, Receiver) pair as
    // the generic channel<T>(cap) form handled in GenericCall.
    // NativeFn cannot construct Rc values, so channel is handled as a special-case
    // Call in eval_expr — this placeholder just exists so lookup doesn't fail.
    e.define("channel", Value::NativeFn {
        name: "channel".into(),
        func: |_args, _line| {
            // Should not be reached: special-cased in eval_expr Call handler.
            Ok(Value::Nil)
        },
    });
    // oneshot/broadcast/watch — interpreter treats them like channel (synchronous simulation).
    // Special-cased in eval_expr Call/GenericCall handlers; placeholders here so lookup succeeds.
    e.define("oneshot", Value::NativeFn {
        name: "oneshot".into(),
        func: |_args, _line| Ok(Value::Nil),
    });
    e.define("broadcast", Value::NativeFn {
        name: "broadcast".into(),
        func: |_args, _line| Ok(Value::Nil),
    });
    e.define("watch", Value::NativeFn {
        name: "watch".into(),
        func: |_args, _line| Ok(Value::Nil),
    });

}

fn register_result_and_args_builtins(e: &mut Env) {
    // ─── Result constructors ──────────────────────────────────────────────────
    // `Ok(v)` and `Err(e)` produce EnumVariant values that pattern-matching
    // (`if let Ok(v) = ...` / `while let Ok(v) = ...`) recognises directly.
    e.define("Ok", Value::NativeFn {
        name: "Ok".into(),
        func: |args, line| {
            if args.len() != 1 {
                return Err(err("Ok() takes exactly 1 argument", line));
            }
            Ok(Value::EnumVariant {
                type_name: "Result".to_string(),
                variant: "Ok".to_string(),
                fields: args.to_vec(),
            })
        },
    });
    e.define("Err", Value::NativeFn {
        name: "Err".into(),
        func: |args, line| {
            if args.len() != 1 {
                return Err(err("Err() takes exactly 1 argument", line));
            }
            Ok(Value::EnumVariant {
                type_name: "Result".to_string(),
                variant: "Err".to_string(),
                fields: args.to_vec(),
            })
        },
    });

    // args() — returns the user-facing CLI arguments for the script.
    // For `boring run file.br -- arg1 arg2`, returns [arg1, arg2].
    // For `boring run file.br arg1 arg2`, returns [arg1, arg2] (skips boring + file).
    e.define("args", Value::NativeFn {
        name: "args".into(),
        func: |_args, _line| {
            let all: Vec<String> = std::env::args().collect();
            let argv: Vec<Value> = if let Some(sep) = all.iter().position(|s| s == "--") {
                all.into_iter().skip(sep + 1).map(Value::Str).collect()
            } else {
                // boring run file.br → user args start at index 3
                all.into_iter().skip(3).map(Value::Str).collect()
            };
            Ok(Value::Array(argv.into()))
        },
    });

    // raw_args() — every CLI argument passed to this program, with no `--`
    // filtering. Unlike args() (meant for a script to read its own pass-through
    // arguments), this is for programs that parse their own command-line syntax
    // — e.g. the self-hosted Boring interpreter, which needs to see `--gpu`,
    // the file path, and `--` all at once instead of having `--` swallow them.
    e.define("raw_args", Value::NativeFn {
        name: "raw_args".into(),
        func: |_args, _line| {
            let all: Vec<String> = std::env::args().collect();
            // boring run file.br → this program's own args start at index 3.
            // A lone leading `--` is dropped if present — it only exists to
            // escape `boring run`'s single-positional-file parser (e.g.
            // `boring run main.br -- --gpu v100 script.br`); everything after
            // it passes through untouched, with no further `--` filtering.
            let mut rest: Vec<String> = all.into_iter().skip(3).collect();
            if rest.first().map(|s| s == "--").unwrap_or(false) {
                rest.remove(0);
            }
            let argv: Vec<Value> = rest.into_iter().map(Value::Str).collect();
            Ok(Value::Array(argv.into()))
        },
    });

    // ord(c) — Unicode codepoint of the first character of a string
    e.define("ord", Value::NativeFn {
        name: "ord".into(),
        func: |args, line| {
            let s = match args.iter().next() {
                Some(Value::Str(s)) => s,
                _ => return Err(err("ord: expected a string argument", line)),
            };
            match s.chars().next() {
                Some(c) => Ok(Value::Int(c as i64)),
                None => Err(err("ord: empty string", line)),
            }
        },
    });

    // chr(code) — string containing the single character for a Unicode codepoint
    e.define("chr", Value::NativeFn {
        name: "chr".into(),
        func: |args, line| {
            let n = match args.iter().next() {
                Some(Value::Int(n)) => *n,
                _ => return Err(err("chr: expected an int argument", line)),
            };
            match char::from_u32(n as u32) {
                Some(c) => Ok(Value::Str(c.to_string())),
                None => Err(err(format!("chr: invalid codepoint {}", n), line)),
            }
        },
    });

    // exit(code) — terminate the process with the given exit code
    e.define("exit", Value::NativeFn {
        name: "exit".into(),
        func: |args, _line| {
            let code = match args.iter().next() {
                Some(Value::Int(n)) => *n as i32,
                _ => 0,
            };
            std::process::exit(code);
        },
    });
}

fn register_misc_globals(e: &mut Env) {
    e.define("PI",  Value::Float(std::f64::consts::PI));
    e.define("E",   Value::Float(std::f64::consts::E));
    e.define("INF", Value::Float(f64::INFINITY));
    e.define("NAN", Value::Float(f64::NAN));

    // ─── Native Rust types ────────────────────────────────────────────────────
    // These are pre-registered without any `use` statement.
    // Collection types map to boring's primitive equivalents at runtime;
    // smart-pointer and utility types produce opaque Objects.
    //
    // Collections — map to boring native values:
    //   HashMap / BTreeMap      →  {K=V}  (Dict)
    //   HashSet / BTreeSet      →  {T}    (Set)
    //   Vec / VecDeque          →  [T]    (Array)
    //   String                  →  string
    for name in &["HashMap", "BTreeMap", "HashSet", "BTreeSet",
                  "Vec", "VecDeque", "String"] {
        e.define(name, Value::RustType { name: name.to_string() });
    }
    // Smart pointers and utility types — opaque constructors:
    //   Box, Rc, Arc            →  transparent wrappers (opaque at interp level)
    //   Option                  →  boring uses T? — available for Rust-fluent compat
    //   Result                  →  boring uses throws — available for Rust-fluent compat
    for name in &["Box", "Rc", "Arc", "Option", "Result"] {
        e.define(name, Value::RustType { name: name.to_string() });
    }
    // Time types — opaque in the interpreter (no real async runtime).
    //   Duration.fromSecs(5)    →  RustType (ignored by wait/timeout stubs)
    //   Instant.now()           →  RustType (opaque deadline; ignored by timeout stubs)
    for name in &["Duration", "Instant"] {
        e.define(name, Value::RustType { name: name.to_string() });
    }
}

fn register_stdlib(env: &EnvRef) {
    let mut e = env.borrow_mut();
    register_io_and_diagnostics_builtins(&mut e);
    register_numeric_conversion_builtins(&mut e);
    register_string_and_math_builtins(&mut e);
    register_concurrency_builtins(&mut e);
    register_result_and_args_builtins(&mut e);
    register_misc_globals(&mut e);
}

// ─── Interpreter ─────────────────────────────────────────────────────────────

pub struct Interpreter {
    /// GPU simulation profile — used by `GpuDevice` property evaluation.
    pub gpu_profile: gpu_profile::GpuProfile,
    pub global: EnvRef,
    pub traits: HashMap<String, TraitDecl>,
    pub enums: HashMap<String, EnumDecl>,  // enum declarations keyed by name
    pub aliases: HashMap<String, Type>,   // user-defined + built-in type aliases
    pub search_paths: Vec<PathBuf>,
    pub loaded: HashSet<PathBuf>,
    pub task_context: bool,  // true at top-level and inside task fns
    /// True while inside a `kernel:` execution block.
    /// Enables implicit `.wait()` on KernelHandle bare expressions.
    pub(crate) kernel_context: bool,
    /// Defer stack: one entry per active function call frame.
    /// Each entry is a list of deferred statement blocks (inner Vec = one `defer:` body).
    /// Blocks are pushed in order and executed in reverse (LIFO) on function exit.
    pub(crate) defer_stack: Vec<Vec<Vec<Stmt>>>,
    /// Stack of type-parameter bindings for generic function/method calls.
    /// Each frame maps type param name → concrete resolved Type.
    /// Frames are pushed on entry to a generic scope and popped on exit.
    pub(crate) type_param_stack: Vec<HashMap<String, Type>>,
    /// True if the currently executing method was declared as `def` (mutating).
    /// False if declared as `req` (non-mutating). Used to enforce transient-field write rules.
    pub(crate) current_method_mutating: bool,
    /// True while executing an `init` body. Bypasses the field immutability check so that
    /// constructors can assign to `let` fields (which are immutable after construction).
    pub(crate) in_init_body: bool,
    /// Runtime store for `type var` values: "StructName::var_name" → Value.
    /// `type let` values are stored here too (immutability enforced by the interpreter).
    pub(crate) type_var_store: HashMap<String, Value>,
    /// True while executing a `type set` body — prevents re-invoking the setter
    /// when the body itself assigns to the same type var (e.g. `Counter.count = v`).
    pub(crate) in_type_setter: bool,
    /// True while executing a stream function body. `yield` statements push to
    /// `stream_yields` as a side effect instead of emitting a Signal::Yield.
    pub(crate) in_stream: bool,
    /// Accumulated yielded values for the current stream function call.
    pub(crate) stream_yields: Vec<Value>,
    /// Arguments forwarded to the user's script (for `args()` built-in).
    pub user_args: Vec<String>,
    /// Final values of `var` params after the most recent function call.
    /// Keyed by param name. Cleared and repopulated by every call_fn invocation.
    /// Read by the call site to write back mutated values to caller variables.
    pub(crate) last_var_params: HashMap<String, Value>,
    /// Block-shared storage for the currently-executing kernel's `'sync` fields
    /// (field name -> shared, mutex-protected backing array), populated only on
    /// the per-thread `Interpreter` instances `run_kernel_parallel` builds for a
    /// kernel that has at least one `'sync` field. Reads/writes to a name in
    /// this map bypass the normal `Env` lookup entirely (see `eval_expr`'s
    /// `ExprKind::Index` case and `assign`'s), so every real OS thread in the
    /// same block observes the same underlying storage — unlike every other
    /// kernel field, which each thread runs against its own independent copy.
    pub(crate) sync_fields: eval_gpu::SyncFieldsMap,
    /// Barrier shared by every thread in the current kernel block, sized to
    /// the block's thread count. `sync` (parsed as `Stmt::Comment("sync")`)
    /// calls `.wait()` on this when present instead of being a no-op — only
    /// set on kernels with `'sync` fields (see `sync_fields`).
    pub(crate) kernel_barrier: Option<std::sync::Arc<std::sync::Barrier>>,
    /// This thread's index within its warp (`gpu.warp.lane`) — always set to
    /// a real value inside a kernel thread (see `eval_gpu::run_one_kernel_thread`),
    /// `0` outside kernel execution.
    pub(crate) warp_lane: usize,
    /// Number of lanes actually participating in this thread's warp-group
    /// (`<= eval_gpu::WARP_SIZE`; less than `WARP_SIZE` only for a block whose
    /// thread count isn't a multiple of `WARP_SIZE`). Used to bound
    /// `gpu.warp.shuffle_*`'s target-lane reads to real participants.
    pub(crate) warp_active_lanes: usize,
    /// Barrier shared by every thread in this thread's warp-group.
    /// `gpu.warp.sync()` waits on this; `None` outside a kernel dispatch that
    /// actually uses `gpu.warp.*` (see `eval_gpu::stmts_use_gpu_warp`).
    pub(crate) warp_barrier: Option<std::sync::Arc<std::sync::Barrier>>,
    /// Shared scratch slots (one per lane, `WARP_SIZE` of them) this thread's
    /// warp-group shuffles values through — `gpu.warp.shuffle_*` writes this
    /// thread's value to `scratch[warp_lane]`, waits on `warp_barrier`, then
    /// reads `scratch[target_lane]`. `None` alongside `warp_barrier`.
    pub(crate) warp_scratch: Option<std::sync::Arc<std::sync::Mutex<Vec<eval_gpu::ThreadValue>>>>,
}

impl Interpreter {
    pub fn new() -> Self {
        let mut aliases: HashMap<String, Type> = HashMap::new();
        // Built-in lowercase aliases
        aliases.insert("int".into(),    Type::Int);
        aliases.insert("uint".into(),   Type::Uint);
        aliases.insert("uint8".into(),  Type::Uint8);
        aliases.insert("int8".into(),    Type::Int8);
        aliases.insert("int16".into(),   Type::Int16);
        aliases.insert("int32".into(),   Type::Int32);
        aliases.insert("int64".into(),   Type::Int64);
        aliases.insert("int128".into(),  Type::Int128);
        aliases.insert("uint16".into(),  Type::Uint16);
        aliases.insert("uint32".into(),  Type::Uint32);
        aliases.insert("uint64".into(),  Type::Uint64);
        aliases.insert("uint128".into(), Type::Uint128);
        aliases.insert("float".into(),  Type::Float);
        aliases.insert("bool".into(),   Type::Bool);
        aliases.insert("string".into(), Type::Qualified(Box::new(Type::Str),   OwnerQual::Shared));
        aliases.insert("str".into(),    Type::Qualified(Box::new(Type::Str),   OwnerQual::Stack));
        // Rust-specific numeric types — distinct real types (each a real Value variant),
        // preserved by name for the transpiler to emit the exact integer width. `isize`/
        // `usize` alias to the bare `int`/`uint` types, which now transpile to isize/usize.
        aliases.insert("i8".into(),    Type::Int8);
        aliases.insert("i16".into(),   Type::Int16);
        aliases.insert("i32".into(),   Type::Int32);
        aliases.insert("i64".into(),   Type::Int64);
        aliases.insert("i128".into(),  Type::Int128);
        aliases.insert("isize".into(), Type::Int);
        aliases.insert("u8".into(),    Type::Uint8);
        aliases.insert("u16".into(),   Type::Uint16);
        aliases.insert("u32".into(),   Type::Uint32);
        aliases.insert("u64".into(),   Type::Uint64);
        aliases.insert("u128".into(),  Type::Uint128);
        aliases.insert("usize".into(), Type::Uint);
        aliases.insert("f32".into(),   Type::Qualified(Box::new(Type::Float), OwnerQual::Stack));
        aliases.insert("f64".into(),   Type::Qualified(Box::new(Type::Float), OwnerQual::Stack));
        // Uppercase base-type aliases — resolve Named("String") etc. to the primitive type so
        // that explicit qualifications like `String'shared`, `Int'copy` work correctly.
        aliases.insert("String".into(), Type::Str);
        aliases.insert("Int".into(),    Type::Int);
        aliases.insert("Uint".into(),   Type::Uint);
        aliases.insert("Uint8".into(),  Type::Uint8);
        aliases.insert("Int8".into(),    Type::Int8);
        aliases.insert("Int16".into(),   Type::Int16);
        aliases.insert("Int32".into(),   Type::Int32);
        aliases.insert("Int64".into(),   Type::Int64);
        aliases.insert("Int128".into(),  Type::Int128);
        aliases.insert("Uint16".into(),  Type::Uint16);
        aliases.insert("Uint32".into(),  Type::Uint32);
        aliases.insert("Uint64".into(),  Type::Uint64);
        aliases.insert("Uint128".into(), Type::Uint128);
        aliases.insert("Float".into(),  Type::Float);
        aliases.insert("Bool".into(),   Type::Bool);
        let global = Env::new_global();
        Self {
            gpu_profile: gpu_profile::GpuProfile::default(),
            global,
            traits: HashMap::new(),
            enums: HashMap::new(),
            aliases,
            search_paths: Vec::new(),
            loaded: HashSet::new(),
            task_context: true,  // top-level is implicitly task context
            kernel_context: false,
            defer_stack: Vec::new(),
            type_param_stack: Vec::new(),
            current_method_mutating: true,  // default to mutating for top-level code
            in_init_body: false,
            type_var_store: HashMap::new(),
            in_type_setter: false,
            in_stream: false,
            stream_yields: Vec::new(),
            user_args: Vec::new(),
            last_var_params: HashMap::new(),
            sync_fields: HashMap::new(),
            kernel_barrier: None,
            warp_lane: 0,
            warp_active_lanes: 0,
            warp_barrier: None,
            warp_scratch: None,
        }
    }

    /// Lightweight interpreter for parallel kernel thread execution.
    /// Copies the parent's type/trait/alias tables and gpu profile but creates a fresh env.
    pub(crate) fn new_for_kernel(
        traits: HashMap<String, TraitDecl>,
        enums: HashMap<String, EnumDecl>,
        aliases: HashMap<String, Type>,
        gpu_profile: gpu_profile::GpuProfile,
    ) -> Self {
        let global = Env::new_global();
        Self {
            gpu_profile,
            global,
            traits,
            enums,
            aliases,
            search_paths: Vec::new(),
            loaded: HashSet::new(),
            task_context: false,
            kernel_context: false,
            defer_stack: Vec::new(),
            type_param_stack: Vec::new(),
            current_method_mutating: true,
            in_init_body: false,
            type_var_store: HashMap::new(),
            in_type_setter: false,
            in_stream: false,
            stream_yields: Vec::new(),
            user_args: Vec::new(),
            last_var_params: HashMap::new(),
            sync_fields: HashMap::new(),
            kernel_barrier: None,
            warp_lane: 0,
            warp_active_lanes: 0,
            warp_barrier: None,
            warp_scratch: None,
        }
    }

    pub fn add_search_path(&mut self, path: PathBuf) {
        self.search_paths.push(path);
    }

    pub fn exec_program(&mut self, program: &Program) -> Result<(), RuntimeError> {
        for item in &program.items {
            if let Err(sig) = self.exec_item(item, Rc::clone(&self.global)) {
                match sig {
                    Signal::Error(e) => return Err(e),
                    Signal::Exception(v) => return Err(RuntimeError {
                        message: format!("unhandled exception: {}", v),
                        line: 0, col: 0, len: 0,
                    }),
                    _ => {}
                }
            }
        }

        // If a `main` function is defined, call it as the entry point.
        // Supports `def main() task:` and `def main() throws:` (in any combination).
        let main_val = self.global.borrow().get("main");
        if let Some(Value::Fn { decl, captured }) = main_val {
            if decl.name == "main" {
                // main() is always treated as a task context — there is no caller that
                // needs to know, so `def main():` works the same as `task main():`.
                let prev_task = self.task_context;
                self.task_context = true;

                let result = self.call_fn(&decl, captured, vec![], 0, decl.throws);

                self.task_context = prev_task;

                match result {
                    Ok(_) => {}
                    Err(Signal::Error(e)) => return Err(e),
                    Err(Signal::Exception(v)) => {
                        // If main is declared `throws`, an uncaught exception is still fatal
                        // at the program level — just give a clear message.
                        return Err(RuntimeError {
                            message: format!("unhandled exception in main: {}", v),
                            line: 0, col: 0, len: 0,
                        });
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    fn check_conformance(&self, decl: &StructDecl) -> Result<(), Signal> {
        // All method names available in the struct (from all sources)
        let available: std::collections::HashSet<&str> = decl.methods.iter()
            .map(|m| m.name.as_str())
            .collect();

        // All trait names claimed by the struct:
        // 1. struct header: `struct Dog as Animal:`  → protocols
        // 2. qualified methods: `def Animal.speak()` → methods with qualifier
        let mut claimed: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for proto in &decl.protocols { claimed.insert(proto); }
        for m in &decl.methods {
            if let Some(q) = &m.qualifier { claimed.insert(q); }
        }

        for trait_name in claimed {
            let Some(trait_decl) = self.traits.get(trait_name) else { continue };
            // A signature is satisfied if the struct provides it OR the trait has a default for it
            let missing: Vec<&str> = trait_decl.signatures.iter()
                .filter(|sig| !available.contains(sig.name.as_str())
                    && !trait_decl.defaults.iter().any(|d| d.name == sig.name))
                .map(|sig| sig.name.as_str())
                .collect();
            if !missing.is_empty() {
                return Err(err(
                    format!("struct '{}' does not conform to '{}': missing method(s): {}",
                        decl.name, trait_name, missing.join(", ")),
                    decl.line,
                ));
            }
            // Check that all associated types declared in the trait are defined
            for assoc in &trait_decl.assoc_types {
                let def = decl.assoc_type_defs.iter().find(|d| d.name == assoc.name);
                if def.is_none() {
                    return Err(err(
                        format!("struct '{}' does not conform to '{}': missing associated type '{}'",
                            decl.name, trait_name, assoc.name),
                        decl.line,
                    ));
                }
                // If the trait has a constraint (e.g. `type Display as string`), verify the
                // concrete type matches (by comparing the resolved display name).
                if let (Some(constraint), Some(def)) = (&assoc.constraint, def) {
                    let constraint_name = Self::display_type(constraint);
                    let def_name = Self::display_type(&def.ty);
                    if constraint_name != def_name {
                        return Err(err(
                            format!("struct '{}': associated type '{}' must be '{}' (as required by '{}'), but got '{}'",
                                decl.name, assoc.name, constraint_name, trait_name, def_name),
                            def.line,
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn exec_use(&mut self, decl: &UseDecl, env: EnvRef) -> Result<(), Signal> {
        // Convert `use a.b.c` → relative path `a/b/c.br`
        let rel: PathBuf = decl.path.iter().collect::<PathBuf>().with_extension("br");

        // Search in registered paths
        let mut found: Option<PathBuf> = None;
        for base in &self.search_paths {
            let candidate = base.join(&rel);
            if candidate.exists() {
                found = Some(candidate);
                break;
            }
        }

        let abs = match found {
            Some(p) => p,
            // No .br file found — treat as a native Rust module (pass-through for the
            // transpiler; the interpreter simply ignores it since native symbols are not
            // available at interpretation time).
            None => return Ok(()),
        };

        // Circular / duplicate import guard
        let canonical = abs.canonicalize().map_err(|e| {
            err(format!("cannot resolve '{}': {}", abs.display(), e), decl.line)
        })?;
        if self.loaded.contains(&canonical) {
            return Ok(());
        }
        self.loaded.insert(canonical.clone());

        // Add the module's directory to search paths for transitive `use`
        if let Some(dir) = canonical.parent() {
            if !self.search_paths.contains(&dir.to_path_buf()) {
                self.search_paths.push(dir.to_path_buf());
            }
        }

        let source = std::fs::read_to_string(&canonical).map_err(|e| {
            err(format!("cannot read '{}': {}", canonical.display(), e), decl.line)
        })?;

        let tokens = crate::lexer::lex(&source).map_err(|e| {
            err(format!("lex error in '{}': {}", canonical.display(), e), decl.line)
        })?;

        let program = crate::parser::parse(tokens).map_err(|e| {
            err(format!("parse error in '{}': {}", canonical.display(), e), decl.line)
        })?;
        // Every other entry point (`run_file`, `parse_and_merge_program`, the
        // transpile/build paths) runs `desugar_labeled_array` on a program
        // right after parsing, before anything else sees it — this runtime
        // `use` loader is the interpreter's own separate parse path for
        // imported modules and was missing that step, so a dynamic-shape
        // `[T, width, height]` kernel field defined in a `use`-imported module
        // (as opposed to the entry file itself) reached `exec_item` still
        // holding an un-desugared `Type::LabeledArray`, which then failed at
        // eval time (confirmed via a real crash migrating whisper-boring's
        // audio_gpu.br — see `dynamic_shape_kernel_field_in_a_use_imported_
        // module_works` in `tests_labeled_array.rs`).
        let program = crate::desugar_labeled_array::desugar_labeled_array(program);

        let module_env = Env::child(Rc::clone(&env));
        for item in &program.items {
            self.exec_item(item, Rc::clone(&module_env))?;
        }
        if decl.items.is_empty() {
            // No filter — export every named item (pub or not).
            // Within a project, all symbols from sibling files are accessible;
            // `pub` only controls external-package visibility (like Rust's pub(crate)).
            for item in &program.items {
                if let Some((name, _)) = Self::item_pub_name(item) {
                    if let Some(val) = module_env.borrow().get(name) {
                        // For enum namespaces, also export each variant as a bare name.
                        if let Value::EnumNamespace { ref variants, .. } = val {
                            for (variant_name, variant_val) in variants {
                                env.borrow_mut().define(variant_name, variant_val.clone());
                            }
                        }
                        env.borrow_mut().define(name, val);
                    }
                }
            }
        } else {
            // Selective import — only export the listed names.
            let module_name = decl.path.join(".");
            for item_name in &decl.items {
                // Must be declared pub in the module.
                let is_pub = program.items.iter().any(|item| {
                    matches!(Self::item_pub_name(item), Some((n, true)) if n == item_name)
                });
                if !is_pub {
                    return Err(err(
                        format!("'{}' is not exported by module '{}'", item_name, module_name),
                        decl.line,
                    ));
                }
                if let Some(val) = module_env.borrow().get(item_name) {
                    env.borrow_mut().define(item_name, val);
                }
            }
        }
        Ok(())
    }

    fn exec_item(&mut self, item: &Item, env: EnvRef) -> Result<(), Signal> {
        match item {
            Item::Use(decl) => self.exec_use_decl(decl, env),
            Item::Fn(decl) => {
                // Check param and return types for unqualified named types
                if let Some(ret) = &decl.return_ty {
                    self.check_type_has_qualifier(ret, decl.line)?;
                }
                for param in &decl.params {
                    if let Some(ty) = &param.ty {
                        self.check_type_has_qualifier(ty, param.line)?;
                    }
                }
                // `native` functions are implemented by the runtime; don't shadow built-ins.
                if !decl.is_native {
                    if let Some(type_name) = &decl.qualifier {
                        // `def TypeName.method()` at top-level — inject into the struct's
                        // method list, exactly like `ext TypeName:` does.
                        let existing = env.borrow().get(type_name);
                        match existing {
                            Some(Value::Struct { decl: mut struct_decl, captured: struct_cap }) => {
                                struct_decl.methods.retain(|m| m.name != decl.name);
                                struct_decl.methods.push(decl.clone());
                                env.borrow_mut().define(type_name, Value::Struct {
                                    decl: struct_decl,
                                    captured: struct_cap,
                                });
                            }
                            _ => {
                                return Err(err(
                                    format!("def {}.{}: unknown struct '{}'", type_name, decl.name, type_name),
                                    decl.line,
                                ));
                            }
                        }
                    } else {
                        let new_fn = Value::Fn { decl: decl.clone(), captured: Rc::clone(&env) };
                        // Check if this name is already defined as a function — if so, upgrade to OverloadedFn.
                        let existing = env.borrow().vars.get(&decl.name).cloned();
                        match existing {
                            Some(Value::Fn { decl: existing_decl, captured: existing_cap }) => {
                                // Conflict check: default params must not create ambiguous overloads.
                                check_overload_conflict_or_exit(&existing_decl, decl, 0);
                                let overloaded = Value::OverloadedFn {
                                    name: decl.name.clone(),
                                    variants: vec![
                                        (existing_decl, existing_cap),
                                        (decl.clone(), Rc::clone(&env)),
                                    ],
                                };
                                env.borrow_mut().define(&decl.name, overloaded);
                            }
                            Some(Value::OverloadedFn { name, mut variants }) => {
                                // Check the new variant against every existing one.
                                for (existing_decl, _) in &variants {
                                    check_overload_conflict_or_exit(existing_decl, decl, 0);
                                }
                                variants.push((decl.clone(), Rc::clone(&env)));
                                env.borrow_mut().define(&decl.name, Value::OverloadedFn { name, variants });
                            }
                            _ => {
                                env.borrow_mut().define(&decl.name, new_fn);
                            }
                        }
                    }
                }
                Ok(())
            }
            Item::Struct(decl) => {
                self.check_conformance(decl)?;
                // Guard: if the first `as` item is a struct name, reject it —
                // use composition + `as Type:` instead.
                if let Some(first_proto) = decl.protocols.first() {
                    let pval = env.borrow().get(first_proto);
                    if matches!(pval, Some(Value::Struct { .. })) {
                        return Err(err(
                            format!(
                                "struct '{}' cannot inherit from struct '{}' — use composition (add a '{}' field) and define 'as {}:' for explicit conversion",
                                decl.name, first_proto, first_proto, first_proto
                            ),
                            decl.line,
                        ));
                    }
                }
                let mut merged = decl.clone();
                // Inject trait default implementations for methods not overridden by the struct.
                for proto in &decl.protocols {
                    if let Some(trait_decl) = self.traits.get(proto.as_str()).cloned() {
                        for default_fn in &trait_decl.defaults {
                            if !merged.methods.iter().any(|m| m.name == default_fn.name) {
                                merged.methods.push(default_fn.clone());
                            }
                        }
                    }
                }
                // Initialise type vars/lets: evaluate default expressions and store them.
                for tv in &decl.type_vars {
                    let key = format!("{}::{}", decl.name, tv.name);
                    let val = self.eval_expr(&tv.default, Rc::clone(&env))?;
                    self.type_var_store.insert(key, val);
                }
                // Register the merged struct
                let val = Value::Struct { decl: merged, captured: Rc::clone(&env) };
                env.borrow_mut().define(&decl.name, val.clone());
                // Also register in global so call_method can find it by type_name regardless of nesting depth.
                if !Rc::ptr_eq(&env, &self.global) {
                    self.global.borrow_mut().define(&decl.name, val);
                }
                // Check that every field type carries a qualifier for heap types
                for field in &decl.fields {
                    self.check_type_has_qualifier(&field.ty, field.line)?;
                }
                // Check method param and return types
                for method in &decl.methods {
                    if let Some(ret) = &method.return_ty {
                        self.check_type_has_qualifier(ret, method.line)?;
                    }
                    for param in &method.params {
                        if let Some(ty) = &param.ty {
                            self.check_type_has_qualifier(ty, param.line)?;
                        }
                    }
                }
                // Check setter param types
                for setter in &decl.setters {
                    self.check_type_has_qualifier(&setter.param_ty, setter.line)?;
                }
                Ok(())
            }
            Item::Enum(decl) => {
                Self::check_enum_field_qualifiers(decl)?;
                let mut variants = HashMap::new();
                for variant in &decl.variants {
                    if variant.fields.is_empty() {
                        variants.insert(
                            variant.name.clone(),
                            Value::EnumVariant {
                                type_name: decl.name.clone(),
                                variant: variant.name.clone(),
                                fields: vec![],
                            },
                        );
                    } else {
                        // Constructor function
                        let type_name = decl.name.clone();
                        let variant_name = variant.name.clone();
                        let field_count = variant.fields.len();
                        // We'll store it as a constructor value
                        variants.insert(
                            variant.name.clone(),
                            Value::EnumVariant {
                                type_name: type_name.clone(),
                                variant: variant_name.clone(),
                                fields: vec![], // will be populated on call
                            },
                        );
                        let _ = field_count; // suppress warning
                    }
                }
                let ns = Value::EnumNamespace {
                    name: decl.name.clone(),
                    variants,
                    methods: decl.methods.clone(),
                    setters: decl.setters.clone(),
                    conversions: decl.conversions.clone(),
                    protocols: decl.protocols.clone(),
                    captured: Rc::clone(&env),
                };
                self.enums.insert(decl.name.clone(), decl.clone());
                // Also define each variant as a bare name so they're accessible from
                // struct methods whose captured env is this module env.
                if let Value::EnumNamespace { ref variants, .. } = ns {
                    for (vname, vval) in variants {
                        env.borrow_mut().define(vname, vval.clone());
                    }
                }
                env.borrow_mut().define(&decl.name, ns.clone());
                if !Rc::ptr_eq(&env, &self.global) {
                    self.global.borrow_mut().define(&decl.name, ns);
                }
                Ok(())
            }
            Item::Trait(decl) => {
                self.traits.insert(decl.name.clone(), decl.clone());
                Ok(())
            }
            Item::Let(stmt) => {
                // Top-level lets are always global; `is_static` here is a no-op
                // (top-level is already global), but we honour `is_pub` as a marker.
                if let Some(ty) = &stmt.ty {
                    self.check_type_has_qualifier(ty, stmt.line)?;
                }
                // Deferred initialisation (`let v` / `var v` without `= expr`).
                // Always mutable so the first branch assignment is allowed.
                let Some(stmt_value) = &stmt.value else {
                    env.borrow_mut().define_mut(&stmt.name, Value::Uninitialized);
                    return Ok(());
                };
                Self::check_no_owned_extract(stmt_value, &env, stmt.line)?;
                let val = self.eval_expr(stmt_value, Rc::clone(&env))?;
                // Apply type coercion if annotation is present (e.g. Int literal → Uint).
                // Also try implicit user-defined `as T:` conversion when needed.
                let val = if let Some(ty) = &stmt.ty {
                    let resolved = self.resolve_type(ty);
                    let coerced = Self::coerce_to_type(val, &resolved);
                    if !self.value_matches_type(&coerced, &resolved) {
                        match self.cast_value(coerced.clone(), &resolved, stmt.line) {
                            Ok(converted) if self.value_matches_type(&converted, &resolved) => converted,
                            _ => {
                                let is_inferred = Self::is_inferred_type(&resolved);
                                let is_concrete = !is_inferred;
                                if is_concrete {
                                    return Err(err(
                                        format!(
                                            "cannot assign {} to '{}': expected {}",
                                            coerced.type_name(),
                                            stmt.name,
                                            Self::display_type(&resolved),
                                        ),
                                        stmt.line,
                                    ));
                                }
                                coerced
                            }
                        }
                    } else { coerced }
                } else { val };
                // Capture copy-ness before `val` is consumed by define()
                let val_is_copy = Self::is_copy_value(&val);
                let is_shared_ty = stmt.ty.as_ref().map(|ty| {
                    matches!(self.resolve_type(ty), Type::Qualified(_, OwnerQual::Shared))
                }).unwrap_or(false);
                let is_actor_ty = stmt.ty.as_ref().map(|ty| {
                    matches!(self.resolve_type(ty), Type::Qualified(_, OwnerQual::Actor | OwnerQual::ActorTask | OwnerQual::Guard | OwnerQual::GuardTask))
                }).unwrap_or(false);
                // Propagate shared/actor status from source when no type annotation.
                let src_var_name = stmt.value.as_ref().and_then(|v| {
                    if let ExprKind::Var(s) = &v.kind { Some(s.clone()) } else { None }
                });
                let src_is_shared = !is_shared_ty && !is_actor_ty && src_var_name.as_ref()
                    .map(|s| env.borrow().is_shared(s.as_str()))
                    .unwrap_or(false);
                let src_is_actor = !is_shared_ty && !is_actor_ty && src_var_name.as_ref()
                    .map(|s| env.borrow().is_actor(s.as_str()))
                    .unwrap_or(false);
                let is_shared_var = stmt.binding.is_mutable() && (is_shared_ty || src_is_shared);
                if is_shared_var {
                    env.borrow_mut().define_shared_mut(&stmt.name, val);
                } else if stmt.binding.is_mutable() {
                    env.borrow_mut().define_mut(&stmt.name, val);
                } else {
                    env.borrow_mut().define(&stmt.name, val);
                }
                if let Some(ty) = &stmt.ty {
                    let resolved = self.resolve_type(ty);
                    if Self::type_has_owned_elems(&resolved) {
                        env.borrow_mut().mark_owned_collection(&stmt.name);
                        Self::invalidate_owned_collection_sources(&resolved, stmt_value, &env);
                    }
                    if Self::type_annotation_is_task_safe(&resolved) {
                        env.borrow_mut().mark_task_safe(&stmt.name);
                    }
                    if matches!(resolved, Type::Qualified(_, OwnerQual::Owned)) {
                        env.borrow_mut().mark_owned_var(&stmt.name);
                    }
                    // Track interior-mutable qualifiers (still gated by `mut`/`var` for `def`
                    // calls, like every other binding — this flag only exempts `'actor`/`'guard`
                    // from the separate `'shared` "no interior mutability" diagnostic below).
                    if is_actor_ty {
                        env.borrow_mut().mark_actor(&stmt.name);
                    }
                    // let T'shared: add to shared_bindings (def methods forbidden, no move on assign).
                    if is_shared_ty {
                        env.borrow_mut().shared_bindings.insert(stmt.name.clone());
                    }
                }
                // Propagate shared/actor to dest when inferred from source.
                if src_is_shared {
                    env.borrow_mut().shared_bindings.insert(stmt.name.clone());
                }
                if src_is_actor {
                    env.borrow_mut().mark_actor(&stmt.name);
                }
                // Move semantics: `let b = a` moves non-copy values by default.
                // Copy types (int, float, bool, nil, void, string) are copied, not moved.
                // Borrow annotations (`T&`) alias rather than move.
                // 'shared/'actor/'guard bindings are reference-counted — assignment is an alias,
                // not a move. Both source and dest remain valid.
                let is_borrow = stmt.ty.as_ref().map(|ty| {
                    matches!(self.resolve_type(ty), Type::Qualified(_, OwnerQual::Borrow | OwnerQual::BorrowMut | OwnerQual::BorrowShared))
                }).unwrap_or(false);
                let src_is_rc_like = src_var_name.as_ref().map(|s| {
                    env.borrow().is_shared(s.as_str()) || env.borrow().is_actor(s.as_str())
                }).unwrap_or(false);
                if !val_is_copy && !is_borrow && !src_is_rc_like {
                    if let Some(v) = &stmt.value {
                        if let ExprKind::Var(src) = &v.kind {
                            env.borrow_mut().set_moved(src.as_str());
                        }
                    }
                }
                Ok(())
            }
            Item::Ext(decl) => {
                // All-native ext on an unknown or opaque (RustType/primitive) target
                // is a documentation-only declaration.  Skip silently — the runtime
                // already provides the methods.
                let is_doc_only = decl.methods.iter().all(|m| m.is_native)
                    && decl.setters.is_empty()
                    && decl.conversions.is_empty()
                    && decl.assoc_type_defs.is_empty();

                // Retrieve the existing struct/enum value
                let existing = env.borrow().get(&decl.type_name);
                let existing = match existing {
                    Some(Value::RustType { .. }) if is_doc_only => return Ok(()),
                    Some(v) => v,
                    None if is_doc_only => return Ok(()),
                    None => return Err(err(
                        format!("ext: unknown type '{}'", decl.type_name),
                        decl.line,
                    )),
                };

                // Handle enum ext
                if let Value::EnumNamespace { name, variants, mut methods, setters: mut enum_setters, mut conversions, mut protocols, captured: enum_captured } = existing {
                    for m in &decl.methods {
                        methods.retain(|existing_m| {
                            existing_m.name != m.name || !params_same_signature(existing_m, m)
                        });
                        methods.push(m.clone());
                    }
                    for s in &decl.setters {
                        enum_setters.retain(|existing_s: &SetDecl| existing_s.name != s.name);
                        enum_setters.push(s.clone());
                    }
                    for conv in &decl.conversions {
                        let conv_ty = format!("{:?}", conv.ty);
                        conversions.retain(|existing| format!("{:?}", existing.ty) != conv_ty);
                        conversions.push(conv.clone());
                    }
                    for trait_name in &decl.traits {
                        if !protocols.contains(trait_name) {
                            protocols.push(trait_name.clone());
                        }
                    }
                    // Verify conformance
                    let all_methods: std::collections::HashSet<&str> = methods.iter().map(|m| m.name.as_str()).collect();
                    for trait_name in &decl.traits {
                        let Some(trait_decl) = self.traits.get(trait_name.as_str()) else { continue };
                        let missing: Vec<&str> = trait_decl.signatures.iter()
                            .filter(|sig| !all_methods.contains(sig.name.as_str()))
                            .map(|sig| sig.name.as_str())
                            .collect();
                        if !missing.is_empty() {
                            return Err(err(
                                format!("ext '{}' as '{}': missing method(s): {}",
                                    decl.type_name, trait_name, missing.join(", ")),
                                decl.line,
                            ));
                        }
                    }
                    let updated = Value::EnumNamespace { name, variants, methods, setters: enum_setters, conversions, protocols, captured: enum_captured };
                    env.borrow_mut().define(&decl.type_name, updated);
                    return Ok(());
                }

                let Value::Struct { decl: mut struct_decl, captured } = existing else {
                    return Err(err(
                        format!("ext: '{}' is not a struct or enum", decl.type_name),
                        decl.line,
                    ));
                };

                // Merge methods: ext overrides an existing method with the SAME signature
                // (same name + same param types), but preserves methods with the same
                // name but different param types (overloads).
                for m in &decl.methods {
                    struct_decl.methods.retain(|existing_m| {
                        existing_m.name != m.name || !params_same_signature(existing_m, m)
                    });
                    struct_decl.methods.push(m.clone());
                }
                // Merge setters
                for s in &decl.setters {
                    struct_decl.setters.retain(|existing_s| existing_s.name != s.name);
                    struct_decl.setters.push(s.clone());
                }
                // Merge as-conversions (keyed by target type display)
                for conv in &decl.conversions {
                    let conv_ty = format!("{:?}", conv.ty);
                    struct_decl.conversions.retain(|existing| format!("{:?}", existing.ty) != conv_ty);
                    struct_decl.conversions.push(conv.clone());
                }
                // Register traits declared in the `as` clause
                for trait_name in &decl.traits {
                    if !struct_decl.protocols.contains(trait_name) {
                        struct_decl.protocols.push(trait_name.clone());
                    }
                }

                // Verify conformance for all traits claimed by this ext
                let all_methods: std::collections::HashSet<&str> = struct_decl.methods.iter()
                    .map(|m| m.name.as_str())
                    .collect();

                let claimed_traits: Vec<String> = decl.traits.to_vec();

                for trait_name in &claimed_traits {
                    let Some(trait_decl) = self.traits.get(trait_name.as_str()) else { continue };
                    let missing: Vec<&str> = trait_decl.signatures.iter()
                        .filter(|sig| !all_methods.contains(sig.name.as_str()))
                        .map(|sig| sig.name.as_str())
                        .collect();
                    if !missing.is_empty() {
                        return Err(err(
                            format!("ext '{}' as '{}': missing method(s): {}",
                                decl.type_name, trait_name, missing.join(", ")),
                            decl.line,
                        ));
                    }
                }

                // Re-register the updated struct
                let updated = Value::Struct { decl: struct_decl, captured };
                env.borrow_mut().define(&decl.type_name, updated);
                Ok(())
            }
            Item::Alias(decl) => {
                self.aliases.insert(decl.name.clone(), decl.ty.clone());
                if decl.newtype {
                    // Register a pass-through constructor: `UserId(42)` → 42 at runtime.
                    // The type system enforcement only matters at compile/transpile time.
                    let ctor_name = decl.name.clone();
                    env.borrow_mut().define(&ctor_name, Value::NativeFn {
                        name: ctor_name.clone(),
                        func: |args, line| {
                            args.first().cloned().ok_or_else(|| Signal::Error(RuntimeError {
                                line,
                                message: "newtype constructor requires one argument".into(), col: 0, len: 0,
                            }))
                        },
                    });
                }
                Ok(())
            }
            Item::Stmt(stmt) => {
                self.exec_stmt(stmt, Rc::clone(&env))?;
                Ok(())
            }
            // `mod name:` — flat scoping: items are executed in the current env.
            // Module structure is preserved in the AST for the Rust transpiler.
            Item::Mod(decl) => {
                for item in &decl.items {
                    self.exec_item(item, Rc::clone(&env))?;
                }
                Ok(())
            }
            Item::Kernel(decl) => {
                self.exec_kernel_decl(decl, env)
            }
        }
    }

}

fn strip_qualifiers(ty: &Type) -> &Type {
    match ty {
        Type::Qualified(inner, _) => strip_qualifiers(inner),
        other => other,
    }
}

fn type_matches(a: &Type, b: &Type) -> bool {
    match (a, b) {
        // Fn subtyping: a non-throwing/non-task fn satisfies a throws/task parameter.
        // req (pure) is a subtype of def (mutating): a pure fn can be passed where mutating is expected.
        // (ret_a, params_a, throws_a, task_a, req_a) satisfies (ret_b, params_b, throws_b, task_b, req_b)
        // when ret/params match, throws_a <= throws_b, task_a <= task_b, req_a >= req_b.
        (Type::Fn(ra, pa, ta, ka, qa), Type::Fn(rb, pb, tb, kb, qb)) => {
            ra == rb && pa == pb
                && (!ta || *tb)   // if a throws, b must accept throws
                && (!ka || *kb)   // if a is task, b must accept task
                && (!qb || *qa)   // if b requires pure (req), a must also be pure
        }
        _ => a == b,
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}


#[cfg(test)]
mod tests;
pub(crate) mod eval_gpu;
