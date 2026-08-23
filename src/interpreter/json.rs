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

//! `fromJson<T>(s)` for `boring run` — a **type-directed** JSON materializer.
//!
//! Syntax parsing is delegated to `serde_json` (`serde_json::from_str::<serde_json::Value>`),
//! which gives a generic JSON tree; everything below then walks that tree against the
//! *Boring* target type (`crate::ast::Type` plus the interpreter's own `EnumDecl`/`StructDecl`
//! tables) and builds a `crate::interpreter::Value`. `serde`'s `Deserialize` derive machinery
//! is deliberately not involved: Boring types are user-declared at Boring-source level and
//! only exist as interpreter data at runtime, so there is no static Rust type to derive into.
//!
//! The goal is 1:1 parity with what `boring build` + real `serde` produce for the same source,
//! because the interpreter is meant to mirror the transpiler's semantics — see
//! docs/interpreter-untagged-enum-fromjson-mismatch.md for the divergence that prompted this
//! (before it, `fromJson<T>` was a no-op that returned its *input string* for every `T`).
//!
//! Supported today: scalars, `string`, `[T]`, `{T}`, `{string=V}`, tuples, `T?`, structs
//! (with `@serde(rename = "...")` / `@serde(rename_all = "...")`), and enums in both
//! `@serde(untagged)` and serde's default externally-tagged shape.
//!
//! Not supported (materialization refuses, so `fromJson` yields the absent optional rather
//! than a silently-wrong value): serde's internally-/adjacently-tagged enum representations
//! (`@serde(tag = "...")`, with or without `content`), `@serde(flatten)`, non-string dict
//! keys, and the `json(v)` *serializer* in the other direction (still a stub — see
//! `eval_expr.rs`'s `json` builtin).

use std::rc::Rc;

use serde_json::Value as Json;

use crate::ast::{Attr, EnumDecl, StructDecl, Type};
use crate::interpreter::{make_object, EnvRef, Interpreter, Value};

// ─── `@serde(...)` attribute helpers ─────────────────────────────────────────

/// Split one raw attribute arg into `(key, value)`.
///
/// Attr args are unstructured token soup rebuilt by the parser's `collect_attr_arg`
/// with no separators, so `@serde(rename = "x")` arrives as the single string
/// `rename="x"` and `@serde(untagged)` as `untagged`. Surrounding quotes on the
/// value are stripped.
fn split_attr_arg(arg: &str) -> (&str, Option<&str>) {
    match arg.find('=') {
        Some(i) => {
            let key = arg[..i].trim();
            let val = arg[i + 1..].trim().trim_matches('"');
            (key, Some(val))
        }
        None => (arg.trim(), None),
    }
}

/// True when any `@serde(...)` attr carries the bare flag `flag` (e.g. `untagged`).
fn has_serde_flag(attrs: &[Attr], flag: &str) -> bool {
    attrs.iter().filter(|a| a.name == "serde").any(|a| {
        a.args.iter().any(|arg| matches!(split_attr_arg(arg), (k, None) if k == flag))
    })
}

/// The value of a `key = "..."` pair inside any `@serde(...)` attr.
fn serde_str_arg(attrs: &[Attr], key: &str) -> Option<String> {
    for a in attrs.iter().filter(|a| a.name == "serde") {
        for arg in &a.args {
            if let (k, Some(v)) = split_attr_arg(arg) {
                if k == key {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}

/// Apply serde's `rename_all` case conversion to one identifier.
///
/// Mirrors `serde_derive`'s own `RenameRule` table. The transpiler never needs this —
/// it emits `#[serde(rename_all = "...")]` verbatim and lets real serde do the work —
/// so there is no existing shared helper to reuse; this is the interpreter-side
/// equivalent, deliberately kept to the rules serde actually recognises (an unknown
/// rule name is a no-op here, where real serde would reject it at compile time).
fn apply_rename_rule(name: &str, rule: &str) -> String {
    // Boring identifiers are conventionally camelCase or snake_case; normalise to
    // lowercase words first so every rule below is a pure join.
    let words = split_identifier_words(name);
    match rule {
        "lowercase" => words.join(""),
        "UPPERCASE" => words.join("").to_uppercase(),
        "PascalCase" => words.iter().map(|w| capitalize(w)).collect::<Vec<_>>().join(""),
        "camelCase" => {
            let mut out = String::new();
            for (i, w) in words.iter().enumerate() {
                if i == 0 { out.push_str(w); } else { out.push_str(&capitalize(w)); }
            }
            out
        }
        "snake_case" => words.join("_"),
        "SCREAMING_SNAKE_CASE" => words.join("_").to_uppercase(),
        "kebab-case" => words.join("-"),
        "SCREAMING-KEBAB-CASE" => words.join("-").to_uppercase(),
        _ => name.to_string(),
    }
}

/// Break an identifier into lowercase words at `_`/`-` separators and camelCase humps.
fn split_identifier_words(name: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    for ch in name.chars() {
        if ch == '_' || ch == '-' {
            if !cur.is_empty() { words.push(std::mem::take(&mut cur)); }
        } else if ch.is_uppercase() && !cur.is_empty() {
            words.push(std::mem::take(&mut cur));
            cur.push(ch.to_ascii_lowercase());
        } else {
            cur.push(ch.to_ascii_lowercase());
        }
    }
    if !cur.is_empty() { words.push(cur); }
    words
}

fn capitalize(w: &str) -> String {
    let mut cs = w.chars();
    match cs.next() {
        Some(c) => c.to_uppercase().collect::<String>() + cs.as_str(),
        None => String::new(),
    }
}

/// The JSON key a struct field / enum variant is (de)serialized under:
/// a per-item `@serde(rename = "...")` wins, then the container's
/// `@serde(rename_all = "...")`, else the Boring spelling verbatim.
fn json_key(name: &str, own_attrs: &[Attr], container_attrs: &[Attr]) -> String {
    if let Some(r) = serde_str_arg(own_attrs, "rename") {
        return r;
    }
    match serde_str_arg(container_attrs, "rename_all") {
        Some(rule) => apply_rename_rule(name, &rule),
        None => name.to_string(),
    }
}

// ─── Type-directed materialization ───────────────────────────────────────────

impl Interpreter {
    /// Entry point for `fromJson<T>(s)`: parse `src` and materialize it as `T`.
    /// Returns `Value::Nil` (the interpreter's "None") on a JSON syntax error or
    /// any shape/type mismatch, matching the transpiler's `serde_json::from_str(..).ok()`.
    pub(crate) fn eval_from_json(&mut self, src: &str, ty: &Type, env: &EnvRef, line: usize) -> Value {
        match serde_json::from_str::<Json>(src) {
            Ok(json) => self.json_to_value(&json, ty, env, line).unwrap_or(Value::Nil),
            Err(_) => Value::Nil,
        }
    }

    /// Alias-resolve `ty` and strip the wrappers that carry no JSON meaning.
    ///
    /// `'stack`/`'heap`/`'shared`/... and `mut` are representation/permission hints only —
    /// the interpreter stores the same `Value` either way. This has to loop: the built-in
    /// alias table already maps `Named("string")`/`Named("int")` onto *qualified* types
    /// (`Type::Qualified(Type::Str, Copy)` and friends), so a single unwrap is not enough
    /// for e.g. `mut string` or a user alias chain.
    fn resolve_json_type(&self, ty: &Type) -> Type {
        let mut cur = self.resolve_type(ty);
        loop {
            match cur {
                Type::Qualified(inner, _) | Type::Mut(inner) => cur = self.resolve_type(&inner),
                other => return other,
            }
        }
    }

    /// Build a `Value` of Boring type `ty` out of the generic JSON node `json`.
    /// `None` means "this JSON does not fit that type" — the caller decides whether
    /// that is a hard failure (top level) or just a rejected untagged-enum candidate.
    pub(crate) fn json_to_value(
        &mut self,
        json: &Json,
        ty: &Type,
        env: &EnvRef,
        line: usize,
    ) -> Option<Value> {
        let ty = self.resolve_json_type(ty);

        match &ty {
            // `T?` — JSON null is the absent case, anything else recurses into `T`.
            Type::Optional(inner) => {
                if json.is_null() {
                    Some(Value::Nil)
                } else {
                    self.json_to_value(json, inner, env, line)
                }
            }

            Type::Bool => json.as_bool().map(Value::Bool),

            Type::Int => json.as_i64().map(Value::Int),
            Type::Int8 => json.as_i64().and_then(|n| i8::try_from(n).ok()).map(Value::Int8),
            Type::Int16 => json.as_i64().and_then(|n| i16::try_from(n).ok()).map(Value::Int16),
            Type::Int32 => json.as_i64().and_then(|n| i32::try_from(n).ok()).map(Value::Int32),
            Type::Int64 => json.as_i64().map(Value::Int64),
            Type::Int128 => json.as_i64().map(|n| Value::Int128(n as i128)),
            Type::Uint => json.as_u64().map(Value::Uint),
            Type::Uint8 => json.as_u64().and_then(|n| u8::try_from(n).ok()).map(Value::Uint8),
            Type::Uint16 => json.as_u64().and_then(|n| u16::try_from(n).ok()).map(Value::Uint16),
            Type::Uint32 => json.as_u64().and_then(|n| u32::try_from(n).ok()).map(Value::Uint32),
            Type::Uint64 => json.as_u64().map(Value::Uint64),
            Type::Uint128 => json.as_u64().map(|n| Value::Uint128(n as u128)),

            // A JSON integer is a valid float (serde accepts `1` for an `f64` field),
            // which is exactly what makes the doc's `JNum(1.0)` output correct.
            Type::Float32 => json.as_f64().map(|n| Value::Float32(n as f32)),
            Type::Float64 => json.as_f64().map(Value::Float64),

            Type::Str => json.as_str().map(|s| Value::Str(s.to_string())),

            Type::Array(elem) | Type::ArrayN(elem, _) => {
                let arr = json.as_array()?;
                let mut out = Vec::with_capacity(arr.len());
                for item in arr {
                    out.push(self.json_to_value(item, elem, env, line)?);
                }
                if let Type::ArrayN(_, n) = &ty {
                    if out.len() != *n { return None; }
                }
                Some(Value::Array(Rc::new(out)))
            }

            Type::Set(elem) => {
                let arr = json.as_array()?;
                let mut out: Vec<Value> = Vec::with_capacity(arr.len());
                for item in arr {
                    let v = self.json_to_value(item, elem, env, line)?;
                    if !out.contains(&v) { out.push(v); }
                }
                Some(Value::Set(out))
            }

            Type::Tuple(elems) => {
                let arr = json.as_array()?;
                if arr.len() != elems.len() { return None; }
                let mut out = Vec::with_capacity(arr.len());
                for (item, ety) in arr.iter().zip(elems.iter()) {
                    out.push(self.json_to_value(item, ety, env, line)?);
                }
                Some(Value::Tuple(out))
            }

            // JSON object keys are always strings, so only a string-keyed dict can
            // round-trip — same restriction real `serde_json` has for map keys.
            Type::Dict(k, v) => {
                if !matches!(self.resolve_json_type(k), Type::Str) { return None; }
                let obj = json.as_object()?;
                let mut pairs = Vec::with_capacity(obj.len());
                for (key, val) in obj {
                    pairs.push((Value::Str(key.clone()), self.json_to_value(val, v, env, line)?));
                }
                Some(Value::Dict(pairs))
            }

            Type::Named(name) => {
                if let Some(decl) = self.enums.get(name).cloned() {
                    return self.json_to_enum(json, &decl, env, line);
                }
                let decl = match env.borrow().get(name) {
                    Some(Value::Struct { decl, .. }) => decl,
                    _ => match self.global.borrow().get(name) {
                        Some(Value::Struct { decl, .. }) => decl,
                        _ => return None,
                    },
                };
                self.json_to_struct(json, &decl, env, line)
            }

            _ => None,
        }
    }

    /// Build a struct instance (an interpreter `Object`) from a JSON object,
    /// honoring `@serde(rename = "...")` per field and `@serde(rename_all = "...")`
    /// on the struct. A field absent from the JSON falls back to its Boring default
    /// (`#[serde(default)]`-like) and is otherwise a mismatch — except an optional
    /// field, which becomes `Nil` the way serde makes a missing `Option<T>` `None`.
    fn json_to_struct(
        &mut self,
        json: &Json,
        decl: &StructDecl,
        env: &EnvRef,
        line: usize,
    ) -> Option<Value> {
        let obj = json.as_object()?;
        let mut fields: Vec<(String, Value)> = Vec::with_capacity(decl.fields.len());
        for f in &decl.fields {
            let key = json_key(&f.name, &f.attrs, &decl.attrs);
            let val = match obj.get(&key) {
                Some(j) => self.json_to_value(j, &f.ty, env, line)?,
                None => {
                    if let Some(default_expr) = f.default.clone() {
                        self.eval_expr(&default_expr, Rc::clone(env)).ok()?
                    } else if matches!(self.resolve_json_type(&f.ty), Type::Optional(_)) {
                        Value::Nil
                    } else {
                        return None;
                    }
                }
            };
            fields.push((f.name.clone(), val));
        }
        Some(make_object(decl.name.clone(), fields))
    }

    /// Build an enum value from JSON, in whichever representation the enum's
    /// `@serde(...)` attrs select: `untagged`, or serde's default externally-tagged form.
    fn json_to_enum(
        &mut self,
        json: &Json,
        decl: &EnumDecl,
        env: &EnvRef,
        line: usize,
    ) -> Option<Value> {
        if has_serde_flag(&decl.attrs, "untagged") {
            self.json_to_enum_untagged(json, decl, env, line)
        } else if serde_str_arg(&decl.attrs, "tag").is_some() {
            // TODO: serde's internally-tagged (`@serde(tag = "t")`) and adjacently-tagged
            // (`tag` + `content`) enum representations aren't implemented. Refuse rather
            // than fall through to the externally-tagged reader, which would happily
            // materialize the *wrong* variant and re-create the very interpreter/compiled
            // divergence docs/interpreter-untagged-enum-fromjson-mismatch.md is about.
            None
        } else {
            self.json_to_enum_external(json, decl, env, line)
        }
    }

    /// `#[serde(untagged)]`: try every variant in declaration order and take the first
    /// that materializes — exactly serde's own algorithm (it runs each variant's
    /// `Deserialize` and keeps the first that doesn't error), which is why declaration
    /// order is load-bearing here.
    fn json_to_enum_untagged(
        &mut self,
        json: &Json,
        decl: &EnumDecl,
        env: &EnvRef,
        line: usize,
    ) -> Option<Value> {
        for v in &decl.variants {
            if let Some(fields) = self.try_variant_payload(json, &v.fields, env, line) {
                return Some(Value::EnumVariant {
                    type_name: decl.name.clone(),
                    variant: v.name.clone(),
                    fields,
                });
            }
        }
        None
    }

    /// serde's default (externally-tagged) representation:
    ///   * a field-less variant  → the bare JSON string `"VariantName"`
    ///   * a variant with fields → the single-key object `{"VariantName": payload}`,
    ///     where `payload` is the value itself for one field and an N-element array
    ///     for N > 1.
    fn json_to_enum_external(
        &mut self,
        json: &Json,
        decl: &EnumDecl,
        env: &EnvRef,
        line: usize,
    ) -> Option<Value> {
        if let Some(tag) = json.as_str() {
            let v = decl.variants.iter()
                .find(|v| v.fields.is_empty() && json_key(&v.name, &v.attrs, &decl.attrs) == tag)?;
            return Some(Value::EnumVariant {
                type_name: decl.name.clone(),
                variant: v.name.clone(),
                fields: vec![],
            });
        }
        let obj = json.as_object()?;
        if obj.len() != 1 { return None; }
        let (tag, payload) = obj.iter().next()?;
        let v = decl.variants.iter()
            .find(|v| json_key(&v.name, &v.attrs, &decl.attrs) == tag.as_str())?
            .clone();
        let fields = self.try_variant_payload(payload, &v.fields, env, line)?;
        Some(Value::EnumVariant {
            type_name: decl.name.clone(),
            variant: v.name.clone(),
            fields,
        })
    }

    /// Materialize one variant's payload fields against `json`, or `None` if it doesn't fit.
    ///
    /// * 0 fields — matches JSON `null` only. (A field-less Boring variant transpiles to a
    ///   Rust *unit* variant, and a unit variant under `#[serde(untagged)]` deserializes
    ///   from `null`; this is what makes `JNull()` pick up the doc repro's `null` element.)
    /// * 1 field  — the JSON value *is* the payload.
    /// * N fields — a JSON array of exactly N elements, matched positionally.
    fn try_variant_payload(
        &mut self,
        json: &Json,
        fields: &[crate::ast::VariantField],
        env: &EnvRef,
        line: usize,
    ) -> Option<Vec<Value>> {
        match fields.len() {
            0 => {
                if json.is_null() { Some(vec![]) } else { None }
            }
            1 => Some(vec![self.json_to_value(json, &fields[0].ty, env, line)?]),
            n => {
                let arr = json.as_array()?;
                if arr.len() != n { return None; }
                let mut out = Vec::with_capacity(n);
                for (item, f) in arr.iter().zip(fields.iter()) {
                    out.push(self.json_to_value(item, &f.ty, env, line)?);
                }
                Some(out)
            }
        }
    }
}
