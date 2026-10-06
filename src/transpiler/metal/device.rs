// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// MSL (Metal Shading Language) device code emitter.

use crate::ast::*;
use crate::transpiler::helpers::{
    reachable_free_fns,
    labeled_array_at_index, labeled_array_dim_literal,
    first_loop_index,
};

#[cfg(test)]
pub(super) fn emit_device_msl(program: &Program) -> String {
    emit_device_msl_with_errors(program).0
}

pub(super) fn emit_device_msl_with_errors(program: &Program) -> (String, Vec<crate::transpiler::TranspileError>) {
    let mut e = DeviceEmitter::new();
    e.emit_program(program);
    (e.out, e.errors)
}

// ── Reserved-word-safe identifiers ────────────────────────────────────────────
//
// MSL's builtin scalar type names (`half`, `float`, `int`, ...) live in the same
// namespace as ordinary identifiers -- unlike Rust, where `f32`/`i64`/etc. can never
// collide with a variable name. An ordinary Boring identifier with no special
// meaning in the language (a kernel field or a `def()`-body local named `half`,
// motivated by a real RoPE positional-encoding kernel -- see CHANGELOG.md) can
// therefore collide with an MSL builtin type, producing a confusing MSL *parse*
// error at runtime (inside `newLibraryWithSource`) rather than a Boring-level
// error -- `boring build --target metal` itself always reports success, since
// this backend never parses the MSL it emits. See docs/metal-backend.md's
// "Naming restrictions" section.

/// MSL builtin scalar type names (`<metal_stdlib>`'s own fixed-width aliases,
/// plus the two 64-bit typedefs `emit_program` itself always emits at the top
/// of every generated file -- `int64_t`/`uint64_t` are just as reserved as any
/// other builtin once declared).
const MSL_SCALAR_TYPES: &[&str] = &[
    "bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong",
    "half", "float", "double", "void",
    "size_t", "ptrdiff_t", "intptr_t", "uintptr_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "atomic_int", "atomic_uint", "atomic_bool", "atomic_float", "atomic_ulong",
];

/// Real C++14/MSL reserved keywords not already covered by `MSL_SCALAR_TYPES`
/// above -- control flow, storage-class specifiers, and MSL's own
/// address-space/function-type qualifiers (`device`, `threadgroup`, `kernel`, ...),
/// which this backend's own generated signatures already treat as reserved.
const MSL_KEYWORDS: &[&str] = &[
    "and", "and_eq", "alignas", "alignof", "asm", "auto", "bitand", "bitor",
    "break", "case", "catch", "class", "compl", "const", "constexpr",
    "const_cast", "continue", "decltype", "default", "delete", "do",
    "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false",
    "for", "friend", "goto", "if", "inline", "mutable", "namespace", "new",
    "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq",
    "private", "protected", "public", "register", "reinterpret_cast",
    "return", "signed", "sizeof", "static", "static_assert", "static_cast",
    "struct", "switch", "template", "this", "thread_local", "throw", "true",
    "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
    "virtual", "volatile", "wchar_t", "char16_t", "char32_t", "while", "xor",
    "xor_eq",
    "kernel", "vertex", "fragment", "constant", "device", "thread",
    "threadgroup", "threadgroup_imageblock", "ray_data", "object_data",
    "patch_control_point", "main",
];

/// `half4`, `float3`, `int2x2`, ... -- MSL's vector/matrix type names, built
/// from the scalar bases that actually support them (MSL has no `double`
/// vectors/matrices, and only `half`/`float` have matrix forms).
fn is_msl_vector_or_matrix_type(name: &str) -> bool {
    const VECTOR_BASES: &[&str] = &[
        "bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "half", "float",
    ];
    const MATRIX_BASES: &[&str] = &["half", "float"];
    for base in VECTOR_BASES {
        for n in 2..=4 {
            if name == format!("{base}{n}") { return true; }
        }
    }
    for base in MATRIX_BASES {
        for n in 2..=4 {
            for m in 2..=4 {
                if name == format!("{base}{n}x{m}") { return true; }
            }
        }
    }
    false
}

fn is_msl_reserved(name: &str) -> bool {
    MSL_SCALAR_TYPES.contains(&name) || MSL_KEYWORDS.contains(&name) || is_msl_vector_or_matrix_type(name)
}

/// Renames a Boring identifier that collides with an MSL reserved word so the
/// generated MSL still parses. A trailing underscore is a no-op for every
/// identifier that ISN'T reserved (the overwhelming majority), so this can be
/// applied unconditionally at every point a Boring field/local/parameter name
/// is emitted as MSL text, both at its declaration and at every later
/// reference -- the mangling is a pure function of the name alone, so a
/// declaration and its uses always agree without needing a rename table.
fn msl_safe_ident(name: &str) -> String {
    if is_msl_reserved(name) { format!("{name}_") } else { name.to_string() }
}

struct DeviceEmitter {
    out: String,
    indent: usize,
    current_fields: Vec<KernelFieldDecl>,
    current_kernel: String,
    /// Auto-barrier mode: true when the kernel `def` has no explicit `sync` statements.
    /// When false (manual mode), the developer owns all barriers.
    auto_sync: bool,
    // Top-level scalar lets inlined into MSL (not in scope in kernel functions).
    top_level_scalars: std::collections::HashMap<String, String>,
    /// Declared Boring types of the current function/method body's local `let`/`var`/`mut`
    /// bindings, populated by `Stmt::Let` and cleared at the start of each device
    /// function/method/entry-point body. Best-effort only -- just accurate enough for
    /// `infer_shuffle_operand_type` to tell whether a `gpu.warp.shuffle_*` operand is
    /// Boring's default `int`/`uint` (MSL `int64_t`/`uint64_t`), which needs the
    /// int32-round-trip cast `gpu_warp_shuffle_msl` applies (see its doc comment).
    locals: std::collections::HashMap<String, Type>,
    /// True while emitting the body of a `void`-returning device function/method --
    /// the tail statement of such a body must stay a bare expression statement
    /// (nothing to return), unlike a non-void function's tail expression.
    current_fn_is_void: bool,
    errors: Vec<crate::transpiler::TranspileError>,
}

impl DeviceEmitter {
    fn new() -> Self {
        Self {
            out: String::new(),
            indent: 0,
            current_fields: vec![],
            current_kernel: String::new(),
            auto_sync: false,
            top_level_scalars: std::collections::HashMap::new(),
            locals: std::collections::HashMap::new(),
            current_fn_is_void: true,
            errors: Vec::new(),
        }
    }

    fn line(&mut self, s: &str) {
        let ind = "    ".repeat(self.indent);
        self.out.push_str(&ind);
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn blank(&mut self) { self.out.push('\n'); }

    fn emit_program(&mut self, program: &Program) {
        // Pre-pass: collect top-level scalar lets for inlining in MSL kernel bodies.
        for item in &program.items {
            if let Item::Let(s) = item {
                if let Some(val) = &s.value {
                    let is_scalar = crate::transpiler::helpers::is_scalar_let_value(val, s.ty.as_ref());
                    if is_scalar {
                        let rhs = self.expr(val);
                        self.top_level_scalars.insert(s.name.clone(), rhs);
                    }
                }
            }
        }

        self.line("// Generated by boring build --target metal.");
        self.blank();
        self.line("#include <metal_stdlib>");
        self.line("#include <metal_simdgroup_matrix>");
        self.line("using namespace metal;");
        self.blank();
        // 64-bit integer aliases for MSL
        self.line("typedef long     int64_t;");
        self.line("typedef ulong    uint64_t;");
        self.blank();
        // Boring built-in Dimension type (mirrors host-side struct)
        self.line("struct Dimension { uint width; uint height; };");
        self.blank();

        // Emit free functions as device helpers callable from any kernel --
        // but only those actually reachable from kernel code. Free functions
        // are ordinary Boring functions shared with the host (CPU) build, and
        // routinely use dynamic-array / heap constructs (`[float]` growable
        // arrays, `.push`, etc.) that have no MSL equivalent (MSL forbids
        // pointer params without an explicit address space and has no heap
        // allocator in device code). Emitting every free function
        // unconditionally — regardless of whether any kernel calls it — used
        // to make the generated kernels/main.metal fail to compile as soon as
        // the program had ANY host-only helper with this shape, even if no
        // kernel ever touched it. Restrict emission to the transitive closure
        // of functions called from kernel entry points/methods instead.
        let reachable = reachable_free_fns(program);
        for item in &program.items {
            if let Item::Fn(decl) = item {
                if decl.qualifier.is_none() && !decl.task && reachable.contains(&decl.name) {
                    self.emit_free_device_fn(decl);
                    self.blank();
                }
            }
        }

        for item in &program.items {
            if let Item::Kernel(decl) = item {
                self.emit_kernel_decl(decl);
            }
        }
    }

    fn emit_free_device_fn(&mut self, decl: &crate::ast::FnDecl) {
        let ret = decl.return_ty.as_ref().map(msl_type).unwrap_or_else(|| "void".into());
        let params: Vec<String> = decl.params.iter().map(|p| {
            let name = msl_safe_ident(&p.name);
            match p.ty.as_ref() {
                Some(ty) => msl_free_fn_param_type(ty, &name, p.mutable),
                None => format!("int64_t {}", name),
            }
        }).collect();
        self.line(&format!("inline {} {}({}) {{", ret, decl.name, params.join(", ")));
        self.indent += 1;
        self.current_fn_is_void = ret == "void";
        self.locals.clear();
        for p in &decl.params {
            if let Some(ty) = &p.ty { self.locals.insert(p.name.clone(), ty.clone()); }
        }
        let last_idx = decl.body.len().saturating_sub(1);
        for (i, stmt) in decl.body.iter().enumerate() { self.emit_stmt(stmt, i == last_idx); }
        self.indent -= 1;
        self.line("}");
    }

    fn emit_kernel_decl(&mut self, decl: &KernelDecl) {
        self.current_fields = decl.fields.clone();
        self.current_kernel = decl.name.clone();
        self.line(&format!("// ─── kernel {} ───", decl.name));
        self.blank();

        // Device helper methods.
        for method in &decl.methods {
            if !method.name.is_empty() {
                self.emit_device_fn(&decl.name, method, &decl.fields);
                self.blank();
            }
        }

        // Entry point: `def ()`.
        if let Some(entry) = decl.methods.iter().find(|m| m.name.is_empty()) {
            self.emit_entry_point(decl, entry);
            self.blank();
        }
    }

    fn emit_device_fn(&mut self, kernel: &str, method: &FnDecl, fields: &[KernelFieldDecl]) {
        let ret = method.return_ty.as_ref().map(msl_type).unwrap_or_else(|| "void".into());
        let fn_name = format!("{}_{}", kernel, method.name);
        let mut params = buffer_field_params(fields);
        for p in &method.params {
            let name = msl_safe_ident(&p.name);
            params.push(match p.ty.as_ref() {
                Some(ty) => msl_free_fn_param_type(ty, &name, p.mutable),
                None => format!("int64_t {}", name),
            });
        }
        self.line(&format!("static {} {}({}) {{", ret, fn_name, params.join(", ")));
        self.indent += 1;
        self.current_fn_is_void = ret == "void";
        self.locals.clear();
        for p in &method.params {
            if let Some(ty) = &p.ty { self.locals.insert(p.name.clone(), ty.clone()); }
        }
        let last_idx = method.body.len().saturating_sub(1);
        for (i, stmt) in method.body.iter().enumerate() { self.emit_stmt(stmt, i == last_idx); }
        self.indent -= 1;
        self.line("}");
    }

    fn emit_entry_point(&mut self, decl: &KernelDecl, entry: &FnDecl) {
        // The entry point is always `void` (a GPU kernel entry has no return value).
        self.current_fn_is_void = true;
        self.locals.clear();
        let fn_name = format!("{}_kernel", decl.name);

        // Build parameter list with [[buffer(N)]] and [[threadgroup(N)]] indices.
        let mut buf_idx: u32 = 0;
        let mut tg_idx: u32 = 0;
        let mut params: Vec<String> = Vec::new();

        // 1. 'unified / 'global / 'actor'global / 'actor'unified array (and LabeledArray) fields → device T* [[buffer(N)]]
        for f in &decl.fields {
            match f.qual {
                GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified => {
                    let elem_ty: Option<&Type> = match &f.ty {
                        Type::Array(inner) | Type::ArrayN(inner, _) | Type::ArrayNExpr(inner, _) => Some(inner.as_ref()),
                        ty if ty.as_labeled_array().is_some() => Some(ty.as_labeled_array().unwrap().0),
                        _ => None,
                    };
                    if let Some(inner) = elem_ty {
                        let elem = elem_msl_type(inner);
                        let constness = if matches!(f.binding, FieldBinding::Let) { "const " } else { "" };
                        params.push(format!("device {}{}* {} [[buffer({})]]",
                            constness, elem, msl_safe_ident(&f.name), buf_idx));
                        buf_idx += 1;
                    }
                }
                // 'surface pixel buffers use 32-bit uint (BGRA8Unorm = 4 bytes/pixel)
                GpuQual::Surface => {
                    match &f.ty {
                        Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) => {
                            params.push(format!("device uint* {} [[buffer({})]]", msl_safe_ident(&f.name), buf_idx));
                            buf_idx += 1;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        // 2. 'const fields → constant T* [[buffer(N)]]
        //    - Scalars: param named __name (pointer), deref'd in body as `const T name = *__name;`
        //    - Fixed arrays: param named name (pointer), accessed directly as name[i]
        for f in &decl.fields {
            if matches!(f.qual, GpuQual::Const) {
                let elem = elem_msl_type(&f.ty);
                match &f.ty {
                    Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) => {
                        // Array: use the field name directly — accessed as name[i] in the kernel body.
                        params.push(format!("constant {}* {} [[buffer({})]]", elem, msl_safe_ident(&f.name), buf_idx));
                    }
                    ty if ty.as_labeled_array().is_some() => {
                        // Fixed-shape LabeledArray: same direct-pointer treatment.
                        params.push(format!("constant {}* {} [[buffer({})]]", elem, msl_safe_ident(&f.name), buf_idx));
                    }
                    _ => {
                        // Scalar: param named __name, dereferenced into a local below.
                        params.push(format!("constant {}* __{} [[buffer({})]]", elem, f.name, buf_idx));
                    }
                }
                buf_idx += 1;
            }
        }

        // 3. Scalar 'local fields → constant T* [[buffer(N)]] (passed from host as scalar)
        for f in &decl.fields {
            if matches!(f.qual, GpuQual::Local)
                && !matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _))
                && f.ty.as_labeled_array().is_none() {
                    let ty = msl_type(&f.ty);
                    params.push(format!("constant {}* __{}_init [[buffer({})]]", ty, f.name, buf_idx));
                    buf_idx += 1;
                }
        }

        // 4. Dynamic 'shared array fields → threadgroup T* [[threadgroup(N)]]
        for f in &decl.fields {
            if matches!(f.qual, GpuQual::Actor)
                && matches!(f.ty, Type::Array(_)) {
                    let elem = elem_msl_type(&f.ty);
                    params.push(format!("threadgroup {}* {} [[threadgroup({})]]", elem, msl_safe_ident(&f.name), tg_idx));
                    tg_idx += 1;
                }
        }

        // 5. Built-in position parameters.
        params.push("uint3 __thread_pos [[thread_position_in_threadgroup]]".into());
        params.push("uint3 __block_pos [[threadgroup_position_in_grid]]".into());
        params.push("uint3 __block_dim [[threads_per_threadgroup]]".into());
        params.push("uint3 __grid_dim [[threadgroups_per_grid]]".into());
        // SIMD-group (warp) built-ins — scalars, unlike the vec3 params above.
        // Unconditional, same as the other built-in position params: MSL
        // SIMD-group builtins need no capability/enable step, so there's no
        // cost to always accepting them even on kernels that don't use
        // `gpu.warp.*`.
        params.push("uint __simd_lane_id [[thread_index_in_simdgroup]]".into());
        params.push("uint __simd_size [[threads_per_simdgroup]]".into());

        self.line(&format!("kernel void {}(", fn_name));
        self.indent += 1;
        for (i, p) in params.iter().enumerate() {
            let comma = if i + 1 < params.len() { "," } else { "" };
            self.line(&format!("{}{}", p, comma));
        }
        self.indent -= 1;
        self.line(") {");
        self.indent += 1;

        // Declare static 'shared arrays inside the kernel body.
        for f in &decl.fields {
            if matches!(f.qual, GpuQual::Actor) {
                if let Type::ArrayN(inner, n) = &f.ty {
                    let elem = elem_msl_type(inner);
                    self.line(&format!("threadgroup {} {}[{}];", elem, msl_safe_ident(&f.name), n));
                } else if let Some((elem, _)) = f.ty.as_labeled_array() {
                    // See cuda::device's identical `labeled_array_len()` note:
                    // `None` (a const-generic axis) isn't handled here yet.
                    if let Some(len) = f.ty.labeled_array_len() {
                        self.line(&format!("threadgroup {} {}[{}];", elem_msl_type(elem), msl_safe_ident(&f.name), len));
                    }
                }
            }
        }

        // Declare 'const scalars: deref the constant buffer pointer into a local.
        // Arrays (and LabeledArray) are passed directly as `constant T* name` and need no deref.
        for f in &decl.fields {
            if matches!(f.qual, GpuQual::Const) {
                match &f.ty {
                    Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) => {
                        // Array: accessed directly via name[i] — no deref needed.
                    }
                    ty if ty.as_labeled_array().is_some() => {}
                    _ => {
                        let ty = msl_type(&f.ty);
                        self.line(&format!("const {} {} = *__{};", ty, msl_safe_ident(&f.name), f.name));
                    }
                }
            }
        }

        // Declare 'local arrays inside the kernel body.
        for f in &decl.fields {
            if matches!(f.qual, GpuQual::Local) {
                match &f.ty {
                    Type::ArrayN(inner, n) => {
                        let elem = elem_msl_type(inner);
                        self.line(&format!("{} {}[{}];", elem, msl_safe_ident(&f.name), n));
                    }
                    Type::Array(_) => {}
                    ty if ty.as_labeled_array().is_some() && ty.labeled_array_len().is_some() => {
                        let (elem, _) = ty.as_labeled_array().unwrap();
                        let len = ty.labeled_array_len().unwrap();
                        self.line(&format!("{} {}[{}];", elem_msl_type(elem), msl_safe_ident(&f.name), len));
                    }
                    // A LabeledArray whose length isn't computable (const-generic
                    // axis) falls here rather than into the scalar branch below —
                    // same "not representable, skip" treatment as `Type::Array(_)`.
                    ty if ty.as_labeled_array().is_some() => {}
                    _ => {
                        // Scalar 'local: initialize from the constant buffer parameter.
                        let ty = msl_type(&f.ty);
                        self.line(&format!("{} {} = *__{}_init;", ty, msl_safe_ident(&f.name), f.name));
                    }
                }
            }
        }

        if decl.name.starts_with("BoringTensorDynamicQ8Native") {
            self.emit_dynamic_packed_linear_body(decl.fields.iter().any(|field| field.name == "bias"), "q8_0", true);
            self.indent -= 1;
            self.line("}");
            return;
        }
        if decl.name.starts_with("BoringTensorDynamicQ4KNative") {
            self.emit_dynamic_packed_linear_body(decl.fields.iter().any(|field| field.name == "bias"), "q4_k", !decl.name.contains("ScalarDecode"));
            self.indent -= 1;
            self.line("}");
            return;
        }
        if decl.name.starts_with("BoringTensorDynamicQ6KNative") {
            self.emit_dynamic_packed_linear_body(decl.fields.iter().any(|field| field.name == "bias"), "q6_k", !decl.name.contains("ScalarDecode"));
            self.indent -= 1;
            self.line("}");
            return;
        }
        if decl.name.starts_with("BoringTensorDynamicFloatNative") {
            self.emit_dynamic_float_linear_body(decl.fields.iter().any(|field| field.name == "bias"));
            self.indent -= 1;
            self.line("}");
            return;
        }

        let has_sync_fields = self.current_fields.iter().any(|f| matches!(f.qual, GpuQual::Actor));
        self.auto_sync = has_sync_fields && !body_has_explicit_sync(&entry.body);
        if self.auto_sync {
            // Emit initial write-phase barrier after the first block of statements that
            // write to 'sync fields, before any loop that reads them cross-thread.
            let split = first_loop_index(&entry.body);
            for stmt in &entry.body[..split] { self.emit_stmt(stmt, false); }
            if split < entry.body.len() {
                self.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
            for stmt in &entry.body[split..] { self.emit_stmt(stmt, false); }
        } else {
            for stmt in &entry.body { self.emit_stmt(stmt, false); }
        }
        self.indent -= 1;
        self.line("}");
    }

    fn emit_dynamic_float_linear_body(&mut self, has_bias: bool) {
        self.line("if ((m % 8) == 0 && (n % 8) == 0 && (k % 8) == 0) {");
        self.indent += 1;
        self.line("const uint bp_row = __block_pos.y * 8;");
        self.line("const uint bp_col = __block_pos.x * 8;");
        self.line("simdgroup_float8x8 bp_result = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);");
        self.line("simdgroup_float8x8 bp_a;");
        self.line("simdgroup_float8x8 bp_b;");
        self.line("for (uint bp_k = 0; bp_k < (uint)k; bp_k += 8) {");
        self.indent += 1;
        self.line("simdgroup_load(bp_a, a, (ulong)k, ulong2(bp_k, bp_row), false);");
        self.line("simdgroup_load(bp_b, b, (ulong)k, ulong2(bp_k, bp_col), true);");
        self.line("simdgroup_multiply_accumulate(bp_result, bp_a, bp_b, bp_result);");
        self.indent -= 1;
        self.line("}");
        self.line("simdgroup_store(bp_result, c, (ulong)n, ulong2(bp_col, bp_row), false);");
        if has_bias {
            self.line("simdgroup_barrier(mem_flags::mem_device);");
            self.line("for (uint bp_cell = __simd_lane_id; bp_cell < 64; bp_cell += __simd_size) {");
            self.indent += 1;
            self.line("const uint bp_dr = bp_cell / 8;");
            self.line("const uint bp_dc = bp_cell % 8;");
            self.line("c[(bp_row + bp_dr) * (uint)n + bp_col + bp_dc] += bias[bp_col + bp_dc];");
            self.indent -= 1;
            self.line("}");
        }
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("const int64_t bp_block = (int64_t)__block_pos.x + (int64_t)__block_pos.y * (int64_t)__grid_dim.x;");
        self.line("const int64_t bp_cell = (int64_t)__thread_pos.x + bp_block * (int64_t)__block_dim.x;");
        self.line("if (bp_cell < m * n) {");
        self.indent += 1;
        self.line("const int64_t bp_row = bp_cell / n;");
        self.line("const int64_t bp_col = bp_cell % n;");
        if has_bias {
            self.line("float bp_sum = bias[bp_col];");
        } else {
            self.line("float bp_sum = 0.0f;");
        }
        self.line("for (int64_t bp_k = 0; bp_k < k; ++bp_k) {");
        self.indent += 1;
        self.line("bp_sum += a[bp_row * k + bp_k] * b[bp_col * k + bp_k];");
        self.indent -= 1;
        self.line("}");
        self.line("c[bp_row * n + bp_col] = bp_sum;");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    fn emit_dynamic_packed_linear_body(&mut self, has_bias: bool, format: &str, warp_decode: bool) {
        if format != "q8_0" {
            self.emit_dynamic_k_quant_linear_body(has_bias, format, warp_decode);
            return;
        }
        self.line("if (m == 1) {");
        self.indent += 1;
        self.line("const uint bp_warp = __thread_pos.x / 32;");
        self.line("const uint bp_col = __block_pos.x * 8 + bp_warp;");
        self.line("const uint bp_lane = __simd_lane_id;");
        self.line("float bp_sum = 0.0f;");
        self.line("if (bp_col < (uint)n) {");
        self.indent += 1;
        self.line("for (uint bp_inner = bp_lane; bp_inner < (uint)k; bp_inner += 32) {");
        self.indent += 1;
        self.line("const ulong bp_flat = (ulong)bp_col * (ulong)k + bp_inner;");
        self.line("const ulong bp_block = (bp_flat / 32) * 34;");
        self.line("float bp_scale = 0.0f;");
        self.line("if (bp_lane == 0) {");
        self.indent += 1;
        self.line("const ushort bp_bits = ushort(ushort(b[bp_block]) | (ushort(b[bp_block + 1]) << 8));");
        self.line("bp_scale = float(as_type<half>(bp_bits));");
        self.indent -= 1;
        self.line("}");
        self.line("bp_scale = simd_shuffle(bp_scale, 0);");
        self.line("int bp_q = int(b[bp_block + 2 + bp_lane]);");
        self.line("if (bp_q > 127) bp_q -= 256;");
        self.line("bp_sum += a[bp_inner] * (float(bp_q) * bp_scale);");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line("for (uint bp_offset = 16; bp_offset > 0; bp_offset /= 2) bp_sum += simd_shuffle_xor(bp_sum, bp_offset);");
        self.line("if (bp_lane == 0 && bp_col < (uint)n) {");
        self.indent += 1;
        if has_bias {
            self.line("c[bp_col] = bp_sum + bias[bp_col];");
        } else {
            self.line("c[bp_col] = bp_sum;");
        }
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("threadgroup float bp_a_tile[256];");
        self.line("threadgroup float bp_b_tile[128];");
        self.line("threadgroup float bp_scales[16];");
        self.line("threadgroup float bp_out_tile[512];");
        self.line("const uint bp_warp = __thread_pos.x / 32;");
        self.line("const uint bp_row_fragment = bp_warp / 2;");
        self.line("const uint bp_col_fragment = bp_warp % 2;");
        self.line("const uint bp_row = __block_pos.y * 32;");
        self.line("const uint bp_col = __block_pos.x * 16;");
        self.line("simdgroup_float8x8 bp_result = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);");
        self.line("simdgroup_float8x8 bp_a;");
        self.line("simdgroup_float8x8 bp_b;");
        self.line("for (uint bp_base = 0; bp_base < (uint)k; bp_base += 8) {");
        self.indent += 1;
        self.line("const uint bp_thread = __thread_pos.x;");
        self.line("const uint bp_input_row = bp_row + bp_thread / 8;");
        self.line("const uint bp_input_inner = bp_base + bp_thread % 8;");
        self.line("bp_a_tile[bp_thread] = bp_input_row < (uint)m ? a[(ulong)bp_input_row * (ulong)k + bp_input_inner] : 0.0f;");
        self.line("if (bp_thread < 16) {");
        self.indent += 1;
        self.line("const uint bp_output_col = bp_col + bp_thread;");
        self.line("if (bp_output_col < (uint)n) {");
        self.indent += 1;
        self.line("const ulong bp_flat = (ulong)bp_output_col * (ulong)k + bp_base;");
        self.line("const ulong bp_block = (bp_flat / 32) * 34;");
        self.line("const ushort bp_bits = ushort(ushort(b[bp_block]) | (ushort(b[bp_block + 1]) << 8));");
        self.line("bp_scales[bp_thread] = float(as_type<half>(bp_bits));");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("bp_scales[bp_thread] = 0.0f;");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        self.line("if (bp_thread < 128) {");
        self.indent += 1;
        self.line("const uint bp_dr = bp_thread / 16;");
        self.line("const uint bp_dc = bp_thread % 16;");
        self.line("const uint bp_output_col = bp_col + bp_dc;");
        self.line("const ulong bp_flat = (ulong)bp_output_col * (ulong)k + bp_base + bp_dr;");
        self.line("const ulong bp_block = (bp_flat / 32) * 34;");
        self.line("int bp_q = bp_output_col < (uint)n ? int(b[bp_block + 2 + (bp_flat % 32)]) : 0;");
        self.line("if (bp_q > 127) bp_q -= 256;");
        self.line("bp_b_tile[bp_thread] = float(bp_q) * bp_scales[bp_dc];");
        self.indent -= 1;
        self.line("}");
        self.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        self.line("simdgroup_load(bp_a, bp_a_tile, 8, ulong2(0, bp_row_fragment * 8), false);");
        self.line("simdgroup_load(bp_b, bp_b_tile, 16, ulong2(bp_col_fragment * 8, 0), false);");
        self.line("simdgroup_multiply_accumulate(bp_result, bp_a, bp_b, bp_result);");
        self.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        self.indent -= 1;
        self.line("}");
        self.line("simdgroup_store(bp_result, bp_out_tile + bp_warp * 64, 8, ulong2(0, 0), false);");
        self.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        self.line("for (uint bp_cell = __simd_lane_id; bp_cell < 64; bp_cell += __simd_size) {");
        self.indent += 1;
        self.line("const uint bp_dr = bp_cell / 8;");
        self.line("const uint bp_dc = bp_cell % 8;");
        self.line("const uint bp_output_row = bp_row + bp_row_fragment * 8 + bp_dr;");
        self.line("const uint bp_output_col = bp_col + bp_col_fragment * 8 + bp_dc;");
        self.line("if (bp_output_row < (uint)m && bp_output_col < (uint)n) {");
        self.indent += 1;
        if has_bias {
            self.line("c[(ulong)bp_output_row * (ulong)n + bp_output_col] = bp_out_tile[bp_warp * 64 + bp_cell] + bias[bp_output_col];");
        } else {
            self.line("c[(ulong)bp_output_row * (ulong)n + bp_output_col] = bp_out_tile[bp_warp * 64 + bp_cell];");
        }
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    fn emit_dynamic_k_quant_linear_body(&mut self, has_bias: bool, format: &str, warp_decode: bool) {
        self.line("if (m == 1) {");
        self.indent += 1;
        if warp_decode {
            self.line("const uint bp_lane = __simd_lane_id;");
            self.line("const uint bp_warp = __thread_pos.x / 32;");
            self.line("const ulong bp_cell = (ulong)__block_pos.x * 8 + bp_warp;");
        } else {
            self.line("const ulong bp_cell = (ulong)__thread_pos.x + ((ulong)__block_pos.x + (ulong)__block_pos.y * (ulong)__grid_dim.x) * (ulong)__block_dim.x;");
        }
        self.line("float bp_sum = 0.0f;");
        self.line("if (bp_cell < (ulong)n) {");
        self.indent += 1;
        if warp_decode {
            self.line("for (ulong bp_base = 0; bp_base < (ulong)k; bp_base += 256) {");
            self.indent += 1;
            self.line(&format!("const ulong bp_block = (bp_cell * (ulong)k + bp_base) / 256 * {};", if format == "q6_k" { 210 } else { 144 }));
            self.line(&format!("float bp_d = bp_lane == 0 ? float(as_type<half>(ushort(ushort(b[bp_block + {}]) | (ushort(b[bp_block + {}]) << 8)))) : 0.0f;", if format == "q6_k" { 208 } else { 0 }, if format == "q6_k" { 209 } else { 1 }));
            self.line("bp_d = simd_shuffle(bp_d, 0);");
            if format == "q4_k" {
                self.line("float bp_dm = bp_lane == 0 ? float(as_type<half>(ushort(ushort(b[bp_block + 2]) | (ushort(b[bp_block + 3]) << 8)))) : 0.0f;");
                self.line("bp_dm = simd_shuffle(bp_dm, 0);");
                self.line("for (uint bp_group = 0; bp_group < 8; ++bp_group) {");
                self.indent += 1;
                self.line("const uint bp_chunk = bp_group / 2, bp_half = bp_group % 2;");
                self.line("int bp_scale = 0, bp_min = 0;");
                self.line("if (bp_lane == 0) {");
                self.indent += 1;
                self.line("bp_scale = bp_group < 4 ? int(b[bp_block + 4 + bp_group]) & 63 : (int(b[bp_block + 8 + bp_group]) & 15) | ((int(b[bp_block + bp_group]) >> 6) << 4);");
                self.line("bp_min = bp_group < 4 ? int(b[bp_block + 8 + bp_group]) & 63 : (int(b[bp_block + 8 + bp_group]) >> 4) | ((int(b[bp_block + 4 + bp_group]) >> 6) << 4);");
                self.indent -= 1;
                self.line("}");
                self.line("bp_scale = simd_shuffle(bp_scale, 0); bp_min = simd_shuffle(bp_min, 0);");
                self.line("const int bp_packed = int(b[bp_block + 16 + bp_chunk * 32 + bp_lane]);");
                self.line("const int bp_q = bp_half == 0 ? bp_packed & 15 : (bp_packed >> 4) & 15;");
                self.line("bp_sum += a[bp_base + bp_group * 32 + bp_lane] * (float(bp_q * bp_scale) * bp_d - float(bp_min) * bp_dm);");
                self.indent -= 1;
                self.line("}");
            } else {
                self.line("for (uint bp_group = 0; bp_group < 8; ++bp_group) {");
                self.indent += 1;
                self.line("const uint bp_iteration = bp_group / 4, bp_quarter = bp_group % 4, bp_half = bp_lane / 16;");
                self.line("const ulong bp_ql = bp_block + bp_iteration * 64, bp_qh = bp_block + 128 + bp_iteration * 32;");
                self.line("const int bp_low0 = int(b[bp_ql + bp_lane]), bp_low32 = int(b[bp_ql + bp_lane + 32]);");
                self.line("const int bp_nibble = bp_quarter == 0 ? bp_low0 & 15 : (bp_quarter == 1 ? bp_low32 & 15 : (bp_quarter == 2 ? (bp_low0 >> 4) & 15 : (bp_low32 >> 4) & 15));");
                self.line("const int bp_high = (int(b[bp_qh + bp_lane]) >> (bp_quarter * 2)) & 3;");
                self.line("int bp_scale = bp_lane % 16 == 0 ? int(b[bp_block + 192 + bp_iteration * 8 + bp_half + bp_quarter * 2]) : 0;");
                self.line("bp_scale = simd_shuffle(bp_scale, bp_half * 16); if (bp_scale > 127) bp_scale -= 256;");
                self.line("bp_sum += a[bp_base + bp_group * 32 + bp_lane] * (float(bp_scale * ((bp_nibble | (bp_high << 4)) - 32)) * bp_d);");
                self.indent -= 1;
                self.line("}");
            }
            self.indent -= 1;
            self.line("}");
        } else {
            self.line("for (ulong bp_inner = 0; bp_inner < (ulong)k; ++bp_inner) {");
            self.indent += 1;
            self.line("const ulong bp_flat = bp_cell * (ulong)k + bp_inner;");
            self.emit_k_quant_decode(format, "bp_flat", "bp_weight");
            self.line("bp_sum += a[bp_inner] * bp_weight;");
            self.indent -= 1;
            self.line("}");
        }
        self.indent -= 1;
        self.line("}");
        if warp_decode {
            self.line("for (uint bp_offset = 16; bp_offset > 0; bp_offset /= 2) bp_sum += simd_shuffle_xor(bp_sum, bp_offset);");
            self.line("if (bp_lane == 0 && bp_cell < (ulong)n) {");
        } else {
            self.line("if (bp_cell < (ulong)n) {");
        }
        self.indent += 1;
        if has_bias { self.line("c[bp_cell] = bp_sum + bias[bp_cell];"); } else { self.line("c[bp_cell] = bp_sum;"); }
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        self.line("threadgroup float bp_a_tile[256];");
        self.line("threadgroup float bp_b_tile[128];");
        self.line("threadgroup float bp_out_tile[512];");
        self.line("const uint bp_thread = __thread_pos.x;");
        self.line("const uint bp_warp = bp_thread / 32;");
        self.line("const uint bp_row_fragment = bp_warp / 2;");
        self.line("const uint bp_col_fragment = bp_warp % 2;");
        self.line("const uint bp_row = __block_pos.y * 32;");
        self.line("const uint bp_col = __block_pos.x * 16;");
        self.line("simdgroup_float8x8 bp_result = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);");
        self.line("simdgroup_float8x8 bp_a;");
        self.line("simdgroup_float8x8 bp_b;");
        self.line("for (uint bp_base = 0; bp_base < (uint)k; bp_base += 8) {");
        self.indent += 1;
        self.line("const uint bp_input_row = bp_row + bp_thread / 8;");
        self.line("const uint bp_input_inner = bp_base + bp_thread % 8;");
        self.line("bp_a_tile[bp_thread] = bp_input_row < (uint)m && bp_input_inner < (uint)k ? a[(ulong)bp_input_row * (ulong)k + bp_input_inner] : 0.0f;");
        self.line("if (bp_thread < 128) {");
        self.indent += 1;
        self.line("const uint bp_dr = bp_thread / 16;");
        self.line("const uint bp_dc = bp_thread % 16;");
        self.line("const uint bp_output_col = bp_col + bp_dc;");
        self.line("const uint bp_inner = bp_base + bp_dr;");
        self.line("if (bp_output_col < (uint)n && bp_inner < (uint)k) {");
        self.indent += 1;
        self.line("const ulong bp_flat = (ulong)bp_output_col * (ulong)k + bp_inner;");
        self.emit_k_quant_decode(format, "bp_flat", "bp_weight");
        self.line("bp_b_tile[bp_thread] = bp_weight;");
        self.indent -= 1;
        self.line("} else bp_b_tile[bp_thread] = 0.0f;");
        self.indent -= 1;
        self.line("}");
        self.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        self.line("simdgroup_load(bp_a, bp_a_tile, 8, ulong2(0, bp_row_fragment * 8), false);");
        self.line("simdgroup_load(bp_b, bp_b_tile, 16, ulong2(bp_col_fragment * 8, 0), false);");
        self.line("simdgroup_multiply_accumulate(bp_result, bp_a, bp_b, bp_result);");
        self.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        self.indent -= 1;
        self.line("}");
        self.line("simdgroup_store(bp_result, bp_out_tile + bp_warp * 64, 8, ulong2(0, 0), false);");
        self.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
        self.line("for (uint bp_cell = __simd_lane_id; bp_cell < 64; bp_cell += __simd_size) {");
        self.indent += 1;
        self.line("const uint bp_dr = bp_cell / 8, bp_dc = bp_cell % 8;");
        self.line("const uint bp_output_row = bp_row + bp_row_fragment * 8 + bp_dr;");
        self.line("const uint bp_output_col = bp_col + bp_col_fragment * 8 + bp_dc;");
        self.line("if (bp_output_row < (uint)m && bp_output_col < (uint)n) {");
        self.indent += 1;
        let bias = if has_bias { " + bias[bp_output_col]" } else { "" };
        self.line(&format!("c[(ulong)bp_output_row * (ulong)n + bp_output_col] = bp_out_tile[bp_warp * 64 + bp_cell]{bias};"));
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    fn emit_k_quant_decode(&mut self, format: &str, flat: &str, output: &str) {
        if format == "q6_k" {
            self.line(&format!("const ulong bp_block = ({flat} / 256) * 210;"));
            self.line(&format!("const uint bp_pos = uint({flat} % 256), bp_iteration = bp_pos / 128, bp_within = bp_pos % 128;"));
            self.line("const uint bp_group = bp_within / 32, bp_lane = bp_within % 32, bp_half = bp_lane / 16;");
            self.line("const ulong bp_ql = bp_block + bp_iteration * 64, bp_qh = bp_block + 128 + bp_iteration * 32;");
            self.line("const int bp_low0 = int(b[bp_ql + bp_lane]), bp_low32 = int(b[bp_ql + bp_lane + 32]);");
            self.line("const int bp_nibble = bp_group == 0 ? bp_low0 & 15 : (bp_group == 1 ? bp_low32 & 15 : (bp_group == 2 ? (bp_low0 >> 4) & 15 : (bp_low32 >> 4) & 15));");
            self.line("const int bp_high = (int(b[bp_qh + bp_lane]) >> (bp_group * 2)) & 3;");
            self.line("int bp_subscale = int(b[bp_block + 192 + bp_iteration * 8 + bp_half + bp_group * 2]); if (bp_subscale > 127) bp_subscale -= 256;");
            self.line("const ushort bp_bits = ushort(ushort(b[bp_block + 208]) | (ushort(b[bp_block + 209]) << 8));");
            self.line(&format!("const float {output} = float(bp_subscale * ((bp_nibble | (bp_high << 4)) - 32)) * float(as_type<half>(bp_bits));"));
        } else {
            self.line(&format!("const ulong bp_block = ({flat} / 256) * 144;"));
            self.line(&format!("const uint bp_pos = uint({flat} % 256), bp_chunk = bp_pos / 64, bp_within = bp_pos % 64;"));
            self.line("const uint bp_half = bp_within / 32, bp_lane = bp_within % 32, bp_subblock = bp_chunk * 2 + bp_half;");
            self.line("const ulong bp_scales = bp_block + 4;");
            self.line("const int bp_subscale = bp_subblock < 4 ? int(b[bp_scales + bp_subblock]) & 63 : (int(b[bp_scales + bp_subblock + 4]) & 15) | ((int(b[bp_scales + bp_subblock - 4]) >> 6) << 4);");
            self.line("const int bp_submin = bp_subblock < 4 ? int(b[bp_scales + bp_subblock + 4]) & 63 : (int(b[bp_scales + bp_subblock + 4]) >> 4) | ((int(b[bp_scales + bp_subblock]) >> 6) << 4);");
            self.line("const int bp_packed = int(b[bp_block + 16 + bp_chunk * 32 + bp_lane]);");
            self.line("const int bp_q = bp_half == 0 ? bp_packed & 15 : (bp_packed >> 4) & 15;");
            self.line("const ushort bp_d_bits = ushort(ushort(b[bp_block]) | (ushort(b[bp_block + 1]) << 8));");
            self.line("const ushort bp_dm_bits = ushort(ushort(b[bp_block + 2]) | (ushort(b[bp_block + 3]) << 8));");
            self.line(&format!("const float {output} = float(bp_q * bp_subscale) * float(as_type<half>(bp_d_bits)) - float(bp_submin) * float(as_type<half>(bp_dm_bits));"));
        }
    }

    // ── Statements ────────────────────────────────────────────────────────────

    fn emit_stmt(&mut self, stmt: &Stmt, is_last: bool) {
        match stmt {
            Stmt::Let(s) => {
                let mutable = matches!(s.binding, BindingKind::Mut | BindingKind::Var | BindingKind::Lazy);
                let ty = s.ty.as_ref().map(msl_type).unwrap_or_else(|| "auto".into());
                let kw = if mutable { "" } else { "const " };
                // Track this binding's Boring type (explicit, or a best-effort guess from
                // its initializer) so `infer_shuffle_operand_type` can later tell whether a
                // `gpu.warp.shuffle_*` call on it needs the int32-round-trip cast.
                let inferred_ty = s.ty.clone().or_else(|| match s.value.as_ref().map(|v| &v.kind) {
                    Some(ExprKind::Int(_)) => Some(Type::Int),
                    Some(ExprKind::Float(_)) => Some(Type::Float64),
                    Some(ExprKind::Cast(_, ty)) => Some(ty.clone()),
                    _ => None,
                });
                if let Some(ty) = inferred_ty { self.locals.insert(s.name.clone(), ty); }
                if let Some(Type::ArrayN(inner, n)) = &s.ty {
                    let name = msl_safe_ident(&s.name);
                    self.line(&format!("{} {}[{}];", elem_msl_type(inner), name, n));
                    if let Some(val) = &s.value { self.emit_fixed_array_init(&name, *n, val); }
                    return;
                }
                if let Some(val) = &s.value {
                    let rhs = self.expr(val);
                    self.line(&format!("{}{} {} = {};", kw, ty, msl_safe_ident(&s.name), rhs));
                } else {
                    self.line(&format!("{}{};", kw, ty));
                }
            }
            Stmt::Expr(e) => {
                if crate::checker::tensor::is_tensor_call(e) {
                    let fields = self.current_fields.clone();
                    let dialect = if self.current_kernel.starts_with("BoringTensorHost") {
                        crate::transpiler::tensor::Dialect::MetalNative
                    } else {
                        crate::transpiler::tensor::Dialect::Metal
                    };
                    let source = crate::transpiler::tensor::emit(
                        e, &fields, dialect,
                        |expr| self.expr(expr),
                    );
                    for line in source.lines() { self.line(line); }
                    return;
                }
                match &e.kind {
                    // `print` → silent no-op in Metal kernels (no device-side printf in MSL).
                    ExprKind::Call(callee, _)
                        if matches!(&callee.kind, ExprKind::Var(n) if n == "print") => {}
                    ExprKind::Assign(lhs, rhs) => {
                        // Atomic compound assign on an 'actor'global field.
                        if let Some(line) = self.try_atomic_assign(lhs, rhs) {
                            self.line(&line);
                        } else {
                            let l = self.expr(lhs);
                            let r = self.expr(rhs);
                            self.line(&format!("{} = {};", l, r));
                        }
                    }
                    _ => {
                        let s = self.expr(e);
                        if is_last && !self.current_fn_is_void {
                            self.line(&format!("return {};", s));
                        } else {
                            self.line(&format!("{};", s));
                        }
                    }
                }
            }
            Stmt::Return(r) => {
                if let Some(val) = &r.value {
                    let s = self.expr(val);
                    self.line(&format!("return {};", s));
                } else {
                    self.line("return;");
                }
            }
            Stmt::If(i) => {
                for (idx, (cond, body)) in i.branches.iter().enumerate() {
                    let c = self.expr(cond);
                    if idx == 0 { self.line(&format!("if ({}) {{", c)); }
                    else        { self.line(&format!("}} else if ({}) {{", c)); }
                    self.indent += 1;
                    let last_idx = body.len().saturating_sub(1);
                    for (j, s) in body.iter().enumerate() { self.emit_stmt(s, is_last && j == last_idx); }
                    self.indent -= 1;
                }
                if let Some(else_body) = &i.else_body {
                    self.line("} else {");
                    self.indent += 1;
                    let last_idx = else_body.len().saturating_sub(1);
                    for (j, s) in else_body.iter().enumerate() { self.emit_stmt(s, is_last && j == last_idx); }
                    self.indent -= 1;
                }
                self.line("}");
            }
            Stmt::While(w) => {
                let cond = self.expr(&w.condition);
                self.line(&format!("while ({}) {{", cond));
                self.indent += 1;
                // In auto-sync mode, insert a barrier at the top of each loop iteration
                // so that 'sync writes from the previous iteration are visible to all threads.
                if self.auto_sync && body_accesses_sync_field(&w.body, &self.current_fields) {
                    self.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
                }
                // A loop body statement is never the function's own tail value, even if
                // this `while` is itself the function's last statement.
                for s in &w.body { self.emit_stmt(s, false); }
                self.indent -= 1;
                self.line("}");
            }
            Stmt::For(f) => {
                let var = msl_safe_ident(&f.vars.first().cloned().unwrap_or_else(|| "_i".into()));
                match &f.iterable.kind {
                    ExprKind::Range { start, end, inclusive } => {
                        let lo = self.expr(start);
                        let hi = self.expr(end);
                        let op = if *inclusive { "<=" } else { "<" };
                        self.line(&format!("for (int64_t {var} = {lo}; {var} {op} {hi}; {var}++) {{"));
                    }
                    _ => {
                        let iter = self.expr(&f.iterable);
                        self.line(&format!("/* for {var} in {iter} -- unsupported */"));
                        self.line("{");
                    }
                }
                self.indent += 1;
                if self.auto_sync && body_accesses_sync_field(&f.body, &self.current_fields) {
                    self.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
                }
                for s in &f.body { self.emit_stmt(s, false); }
                self.indent -= 1;
                self.line("}");
            }
            Stmt::Break(_label, _val) => self.line("break;"),
            Stmt::Continue(_label)    => self.line("continue;"),
            // sync → threadgroup memory barrier.
            Stmt::Comment(c) if c == "sync" => {
                self.line("threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
            Stmt::Comment(_) => {}
            _ => { self.line("/* unsupported stmt in kernel */"); }
        }
    }

    /// `'actor'global` or `'actor'unified` — MSL's atomic cast at the call site doesn't
    /// care which `MTLResourceOptions` backs the buffer, only that it's `device` address
    /// space, so both qualifiers take the identical atomic codegen path.
    fn is_atomic_field(&self, name: &str) -> bool {
        self.current_fields.iter().any(|f|
            f.name == name && matches!(f.qual, GpuQual::ActorGlobal | GpuQual::ActorUnified))
    }

    /// Returns the MSL atomic pointer-cast type and the matching scalar cast type
    /// for the *real* element type of the `'actor'global`/`'actor'unified` array
    /// field named `arr_name` — e.g. `("atomic_int", "int")` for an `int32`
    /// element, not the field-width-blind `("atomic_long", "long")` this backend
    /// used to hardcode for every atomic op regardless of the field's actual MSL
    /// type (`elem_msl_type`). A real 8-byte `atomic_long` cast onto a 4-byte
    /// `int32`/`uint32` element reads/writes 4 bytes past the valid allocation
    /// (or reinterprets a neighboring element's bytes as part of the same atomic
    /// word) — GPU memory corruption or a numerically wrong result, not merely a
    /// style nit.
    ///
    /// `None` for an element type with no correct portable MSL atomic (8/16-bit
    /// integers, floats, and the already-unsupported 64/128-bit named widths) —
    /// callers fall back to a `/* ERROR: ... */`-flagged passthrough, the same
    /// "flag it in a comment, don't silently miscompile" convention
    /// `msl_unsupported_width`/`msl_unsupported_f64` above already use for these
    /// types as plain (non-atomic) fields.
    fn atomic_msl_cast(&self, arr_name: &str) -> Option<(&'static str, &'static str)> {
        let field = self.current_fields.iter().find(|f| f.name == arr_name)?;
        match elem_msl_type(&field.ty).as_str() {
            "int"  => Some(("atomic_int", "int")),
            "uint" => Some(("atomic_uint", "uint")),
            // Bare `int`/`uint` (isize/usize) — this backend's one genuinely
            // 64-bit element type (`msl_type` emits `int64_t`/`uint64_t` for it).
            // Preserved as `atomic_long`/`long`, matching this codegen's
            // pre-existing (and, per `try_atomic_method_call`'s own doc comment,
            // still not independently verified against a real Metal compiler)
            // choice for the 64-bit case specifically — unlike the 32-bit case
            // above, that choice was already width-correct.
            "int64_t" | "uint64_t" => Some(("atomic_long", "long")),
            _ => None,
        }
    }

    /// Detect `arr[i] OP= v` on an `'actor'global`/`'actor'unified` field and emit Metal
    /// atomic intrinsic.
    fn try_atomic_assign(&mut self, lhs: &Expr, rhs: &Expr) -> Option<String> {
        let ExprKind::Index(arr, _idx) = &lhs.kind else { return None; };
        let arr_name = match &arr.kind {
            ExprKind::Var(n) => n.clone(),
            _ => return None,
        };
        if !self.is_atomic_field(&arr_name) { return None; }
        let ExprKind::BinOp(op, _lhs_copy, value) = &rhs.kind else { return None; };

        let target = self.expr(lhs);
        let v = self.expr(value);

        let Some((cast_ty, val_ty)) = self.atomic_msl_cast(&arr_name) else {
            return Some(format!(
                "/* ERROR: atomic op on `{}` has no portable MSL atomic for its element type */;",
                arr_name
            ));
        };

        // MSL atomic intrinsics — no atomicSub: use add with negation.
        let intrinsic = match op {
            BinOp::Add    => format!("atomic_fetch_add_explicit((device {}*)&{}, ({})({}), memory_order_relaxed)", cast_ty, target, val_ty, v),
            BinOp::Sub    => format!("atomic_fetch_add_explicit((device {}*)&{}, -({})({}), memory_order_relaxed)", cast_ty, target, val_ty, v),
            BinOp::BitOr  => format!("atomic_fetch_or_explicit((device {}*)&{}, ({})({}), memory_order_relaxed)", cast_ty, target, val_ty, v),
            BinOp::BitAnd => format!("atomic_fetch_and_explicit((device {}*)&{}, ({})({}), memory_order_relaxed)", cast_ty, target, val_ty, v),
            BinOp::BitXor => format!("atomic_fetch_xor_explicit((device {}*)&{}, ({})({}), memory_order_relaxed)", cast_ty, target, val_ty, v),
            _ => return None,
        };
        Some(format!("{};", intrinsic))
    }

    /// Detect `arr[i].min/max/swap/cas(...)` where `arr` is an
    /// `'actor'global`/`'actor'unified` field and emit the corresponding MSL
    /// atomic intrinsic. Handled in expression position — unlike
    /// `try_atomic_assign`'s statement-only compound-assign desugar, these
    /// return the previous value.
    ///
    /// `min`/`max`/`swap` map directly onto MSL's
    /// `atomic_fetch_min_explicit`/`atomic_fetch_max_explicit`/
    /// `atomic_exchange_explicit`, which already return the previous value
    /// like their CUDA/HIP equivalents — same `(device atomic_long*)` cast
    /// pattern already used for `+=`/`-=`/etc. above (`atomic_fetch_min/max_explicit`
    /// on 64-bit `atomic_long` specifically is not independently verified
    /// against a real Metal compiler in this environment — same caveat this
    /// backend's docs already carry elsewhere for untestable-locally MSL
    /// codegen).
    ///
    /// `cas` is a real shape mismatch: MSL's
    /// `atomic_compare_exchange_weak_explicit(object, &expected, desired, ...)`
    /// takes a *pointer* to the expected value (overwritten with the real
    /// current value on failure) and returns a `bool`, not the previous
    /// value directly — unlike CUDA/HIP's `atomicCAS`, which just returns
    /// it. Bridged via a GNU/Clang statement-expression (`({ ... })`,
    /// supported by Metal's Clang-based compiler) so the whole thing is
    /// still usable as a single expression: bind `expected` into a local,
    /// call compare-exchange (ignoring the bool — the local now holds
    /// whichever value actually ends up correct: unchanged if it succeeded,
    /// the real current value if it didn't), and yield that local.
    fn try_atomic_method_call(&mut self, obj: &Expr, method: &str, args_s: &[String]) -> Option<String> {
        let ExprKind::Index(arr, _idx) = &obj.kind else { return None; };
        let arr_name = match &arr.kind {
            ExprKind::Var(n) => n.clone(),
            _ => return None,
        };
        if !matches!(method, "min" | "max" | "swap" | "cas") { return None; }
        let target = self.expr(obj);
        if self.is_atomic_field(&arr_name) {
            let Some((cast_ty, val_ty)) = self.atomic_msl_cast(&arr_name) else {
                return Some(format!(
                    "/* ERROR: atomic op on `{}` has no portable MSL atomic for its element type */",
                    arr_name
                ));
            };
            match (method, args_s) {
                ("min", [v]) => Some(format!(
                    "atomic_fetch_min_explicit((device {}*)&{}, ({})({}), memory_order_relaxed)", cast_ty, target, val_ty, v)),
                ("max", [v]) => Some(format!(
                    "atomic_fetch_max_explicit((device {}*)&{}, ({})({}), memory_order_relaxed)", cast_ty, target, val_ty, v)),
                ("swap", [v]) => Some(format!(
                    "atomic_exchange_explicit((device {}*)&{}, ({})({}), memory_order_relaxed)", cast_ty, target, val_ty, v)),
                ("cas", [expected, new]) => Some(format!(
                    "({{ {} __boring_cas_exp = ({})({}); atomic_compare_exchange_weak_explicit((device {}*)&{}, &__boring_cas_exp, ({})({}), memory_order_relaxed, memory_order_relaxed); __boring_cas_exp; }})",
                    val_ty, val_ty, expected, cast_ty, target, val_ty, new
                )),
                _ => None,
            }
        } else {
            // Not atomic-qualified -- plain read-modify-write, no atomic cast
            // or memory-order needed. Bridged via the same GNU/Clang
            // statement-expression as the atomic `.cas` case above (Metal's
            // compiler is Clang-based) since there's no intrinsic call to
            // lean on for the "return the previous value" contract here.
            match (method, args_s) {
                ("min", [v]) => Some(format!(
                    "({{ auto __old = {t}; {t} = min({t}, ({v})); __old; }})", t = target, v = v)),
                ("max", [v]) => Some(format!(
                    "({{ auto __old = {t}; {t} = max({t}, ({v})); __old; }})", t = target, v = v)),
                ("swap", [v]) => Some(format!(
                    "({{ auto __old = {t}; {t} = ({v}); __old; }})", t = target, v = v)),
                ("cas", [expected, new]) => Some(format!(
                    "({{ auto __old = {t}; if (__old == ({e})) {t} = ({n}); __old; }})",
                    t = target, e = expected, n = new)),
                _ => None,
            }
        }
    }

    /// `a.axis` — read-only shape-query property on a fixed-shape
    /// `LabeledArray` field (dynamic-shape ones are already desugared to a
    /// shadow field before codegen — see `desugar_labeled_array`). `a[width =
    /// w, height = h]`-style indexing is a distinct `ExprKind::LabeledIndex`
    /// node, handled directly in `expr()`'s main match instead of here.
    fn try_labeled_array_field_access(&mut self, obj: &Expr, field_name: &str) -> Option<String> {
        let ExprKind::Var(name) = &obj.kind else { return None; };
        let field = self.current_fields.iter().find(|f| &f.name == name)?;
        let (_, axes) = field.ty.as_labeled_array()?;
        labeled_array_dim_literal(axes, field_name)
    }

    // ── Expressions ───────────────────────────────────────────────────────────

    /// Declared Boring type of a local binding or (via `self.field`) kernel field
    /// named `name`, if tracked -- see `locals`'s doc comment.
    fn local_or_field_type(&self, name: &str) -> Option<Type> {
        self.locals.get(name).cloned()
            .or_else(|| self.current_fields.iter().find(|f| f.name == name).map(|f| f.ty.clone()))
    }

    /// Best-effort Boring type of `expr`, just accurate enough for
    /// `gpu_warp_method_call` to decide whether a `gpu.warp.shuffle_*` operand needs
    /// the int32-round-trip cast Metal's `simd_shuffle*` requires for a 64-bit int
    /// (see that function's doc comment). Resolves a local/`self.field` variable via
    /// `local_or_field_type`, an explicit cast's target type, or a literal; anything
    /// else (a binary op, an arbitrary call, ...) returns `None`, and the caller
    /// leaves the value un-cast -- unchanged from before this inference existed.
    fn infer_shuffle_operand_type(&self, expr: &Expr) -> Option<Type> {
        match &expr.kind {
            ExprKind::Var(name) => self.local_or_field_type(name),
            ExprKind::Field(obj, name) if matches!(&obj.kind, ExprKind::Var(v) if v == "self") => {
                self.local_or_field_type(name)
            }
            ExprKind::Cast(_, ty) => Some(ty.clone()),
            ExprKind::Int(_) => Some(Type::Int),
            ExprKind::Float(_) => Some(Type::Float64),
            _ => None,
        }
    }

    /// Value of an if/elif/else branch body: the trailing expression statement,
    /// or `"0"` if the branch has none.
    fn if_branch_value(&mut self, body: &[Stmt]) -> String {
        body.last().and_then(|s| {
            if let Stmt::Expr(e) = s { Some(self.expr(e)) } else { None }
        }).unwrap_or_else(|| "0".into())
    }

    fn expr(&mut self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::Int(n)   => n.to_string(),
            ExprKind::Float(f) => {
                let s = format!("{}", f);
                if s.contains('.') || s.contains('e') { s } else { format!("{}.0", s) }
            }
            ExprKind::Bool(b)  => if *b { "1".into() } else { "0".into() },
            ExprKind::Str(s)   => format!("\"{}\"", s),
            ExprKind::Nil      => "0".into(),
            ExprKind::Void     => "".into(),
            ExprKind::Var(name) => {
                // A kernel field of the same name shadows the top-level scalar
                // (e.g. `kernel Saxpy: let float alpha` vs. top-level `let alpha = 2.0`
                // in examples/saxpy.br) -- the field is a real local/parameter in the
                // generated device function, so it must win, not the outer literal.
                // Previously unguarded: silently miscompiled `alpha * x[i] + y[i]` to
                // always use the top-level literal instead of the runtime parameter --
                // no compile error, just a wrong-value bug (confirmed via
                // `boring build --target metal examples/saxpy.br`).
                if self.current_fields.iter().any(|f| f.name == *name) {
                    msl_safe_ident(name)
                } else {
                    self.top_level_scalars.get(name).cloned().unwrap_or_else(|| msl_safe_ident(name))
                }
            }

            ExprKind::BinOp(op, lhs, rhs) => {
                let l = self.expr(lhs);
                let r = self.expr(rhs);
                format!("({} {} {})", l, binop_msl(op), r)
            }
            ExprKind::UnaryOp(op, operand) => {
                let v = self.expr(operand);
                format!("({}{})", unaryop_msl(op), v)
            }
            ExprKind::Assign(lhs, rhs) => {
                format!("({} = {})", self.expr(lhs), self.expr(rhs))
            }
            ExprKind::Index(arr, idx) => {
                format!("{}[{}]", self.expr(arr), self.expr(idx))
            }
            ExprKind::LabeledIndex(obj, args) => {
                // Stringify every arg's value first (needs `&mut self`) —
                // only then borrow `self.current_fields` immutably, so the
                // two borrows never overlap.
                let pairs: Vec<(String, String)> = args.iter()
                    .filter_map(|a| a.label.clone().map(|l| (l, self.expr(&a.value))))
                    .collect();
                let resolved = if let ExprKind::Var(name) = &obj.kind {
                    self.current_fields.iter().find(|f| &f.name == name)
                        .and_then(|field| field.ty.as_labeled_array())
                        .and_then(|(_, axes)| labeled_array_at_index(axes, &pairs))
                        .map(|offset| format!("{}[{}]", msl_safe_ident(name), offset))
                } else {
                    None
                };
                resolved.unwrap_or_else(|| "/* unsupported labeled index */".to_string())
            }
            ExprKind::Field(obj, field) => {
                if let Some(msl) = self.try_labeled_array_field_access(obj, field) {
                    msl
                } else {
                    let obj_s = self.expr(obj);
                    map_gpu_field(&obj_s, field)
                }
            }
            ExprKind::Call(callee, args) => {
                let args_s: Vec<String> = args.iter().map(|a| self.expr(&a.value)).collect();
                let fn_s = match &callee.kind {
                    ExprKind::Var(n) => map_builtin_fn(n),
                    _ => self.expr(callee),
                };
                format!("{}({})", fn_s, args_s.join(", "))
            }
            ExprKind::MethodCall(obj, method, args) => {
                let args_s: Vec<String> = args.iter().map(|a| self.expr(&a.value)).collect();
                if is_gpu_warp_receiver(obj) {
                    let operand_ty = args.first().and_then(|a| self.infer_shuffle_operand_type(&a.value));
                    if let Some(msl) = gpu_warp_method_call(method, &args_s, operand_ty.as_ref()) {
                        return msl;
                    }
                }
                if let Some(msl) = self.try_atomic_method_call(obj, method, &args_s) {
                    return msl;
                }
                if matches!(&obj.kind, ExprKind::Var(n) if n == "self") {
                    // `self.method(args)` → the sibling device function this method was
                    // emitted as (see `emit_device_fn`): `KernelName_method(fields..., args)`.
                    let mut all_args = buffer_field_arg_names(&self.current_fields);
                    all_args.extend(args_s);
                    format!("{}_{}({})", self.current_kernel, method, all_args.join(", "))
                } else if let Some(msl) = float_unary_method_msl(method, &self.expr(obj), &args_s) {
                    // Boring's built-in numeric methods (`x.exp()`, `x.sqrt()`, ...) map
                    // directly onto MSL's `metal_stdlib` free functions of the same shape.
                    msl
                } else {
                    // Method calls on a receiver other than `self` have no MSL equivalent
                    // in this minimal device emitter — leave a visible marker rather than
                    // silently discarding the call.
                    let obj_s = self.expr(obj);
                    format!("/* unsupported: {}.{}({}) */", obj_s, method, args_s.join(", "))
                }
            }
            ExprKind::Cast(inner, ty) => {
                format!("(({})({})) ", msl_type(ty), self.expr(inner))
            }
            ExprKind::If(i) => {
                let tail = match &i.else_body {
                    Some(b) => self.if_branch_value(b),
                    None => "0".into(),
                };
                // Nest every branch (not just the first) so an elif chain lowers to
                // `c0 ? t0 : (c1 ? t1 : (c2 ? t2 : else))` instead of collapsing to
                // just the first condition with the else value as fallback.
                let mut acc = tail;
                for (cond, then_body) in i.branches.iter().rev() {
                    let c = self.expr(cond);
                    let t = self.if_branch_value(then_body);
                    acc = format!("({} ? {} : {})", c, t, acc);
                }
                acc
            }
            ExprKind::Range { start, end, .. } => {
                format!("/* range {}..{} */", self.expr(start), self.expr(end))
            }
            _ => {
                self.errors.push(crate::transpiler::TranspileError::at(
                    "expression is not supported in Metal kernel device code", e.line, e.col,
                ));
                "0".into()
            }
        }
    }

    fn emit_fixed_array_init(&mut self, name: &str, n: usize, value: &Expr) {
        match &value.kind {
            ExprKind::Array(values) if values.len() == n => {
                for (i, value) in values.iter().enumerate() {
                    let rhs = self.expr(value);
                    self.line(&format!("{}[{}] = {};", name, i, rhs));
                }
            }
            ExprKind::ArrayFill { value, count } if const_array_count(count) == Some(n) => {
                let rhs = self.expr(value);
                self.line(&format!("for (ulong bp_array_i = 0; bp_array_i < {}; ++bp_array_i) {{ {}[bp_array_i] = {}; }}", n, name, rhs));
            }
            _ => self.errors.push(crate::transpiler::TranspileError::at(
                "fixed-array kernel locals require an N-element array literal or `[value for ..<N]` initializer",
                value.line, value.col,
            )),
        }
    }
}

fn const_array_count(e: &Expr) -> Option<usize> {
    match e.kind { ExprKind::Int(n) if n >= 0 => Some(n as usize), _ => None }
}


// ── Free helpers ──────────────────────────────────────────────────────────────

/// Parameters for device helper functions (no [[attribute]] annotations).
fn buffer_field_params(fields: &[KernelFieldDecl]) -> Vec<String> {
    fields.iter().filter_map(|f| {
        match f.qual {
            GpuQual::Actor | GpuQual::Local => None,
            GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Surface => {
                let elem = elem_msl_type(&f.ty);
                let constness = if matches!(f.binding, FieldBinding::Let) { "const " } else { "" };
                Some(format!("device {}{}* {}", constness, elem, msl_safe_ident(&f.name)))
            }
            GpuQual::Const => {
                let elem = elem_msl_type(&f.ty);
                match &f.ty {
                    Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) => {
                        // Array: direct pointer, no deref — same as in the entry point.
                        Some(format!("constant {}* {}", elem, msl_safe_ident(&f.name)))
                    }
                    ty if ty.as_labeled_array().is_some() => {
                        Some(format!("constant {}* {}", elem, msl_safe_ident(&f.name)))
                    }
                    _ => {
                        // Scalar: passed as __name pointer so helper can access it.
                        Some(format!("constant {}* __{}", elem, f.name))
                    }
                }
            }
        }
    }).collect()
}

/// Argument names to pass at a call site for `buffer_field_params(fields)` — same
/// filtering as that function, but just the bare field names.
fn buffer_field_arg_names(fields: &[KernelFieldDecl]) -> Vec<String> {
    fields.iter().filter_map(|f| {
        match f.qual {
            GpuQual::Actor | GpuQual::Local => None,
            GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Const | GpuQual::Surface => {
                Some(msl_safe_ident(&f.name))
            }
        }
    }).collect()
}

/// MSL has no native 64/128-bit integer type (Apple GPUs historically lack native 64-bit
/// int ALU ops) — flag it in a comment rather than silently emitting a type that doesn't
/// exist in MSL (the previous behavior for `Uint8`, which fell through to `void*`).
fn msl_unsupported_width(name: &str, fallback: &str) -> String {
    format!("{} /* ERROR: `{}` is not supported on --target metal (MSL has no 64/128-bit integers) */", fallback, name)
}

/// MSL has no native `double` (Apple GPUs don't support 64-bit float ALU ops) — same
/// "flag it in a comment" convention as `msl_unsupported_width` above, with its own
/// message rather than that function's integer-specific wording
/// (docs/float-width-types.md §6).
fn msl_unsupported_f64(fallback: &str) -> String {
    format!("{} /* ERROR: `float64` is not supported on --target metal (MSL has no native double — use float32) */", fallback)
}

fn msl_type(ty: &Type) -> String {
    match ty {
        Type::Int              => "int64_t".into(),
        Type::Uint             => "uint64_t".into(),
        // MSL natively supports char/uchar (8-bit), short/ushort (16-bit), int/uint (32-bit).
        Type::Uint8             => "uchar".into(),
        Type::Int8               => "char".into(),
        Type::Int16              => "short".into(),
        Type::Uint16             => "ushort".into(),
        Type::Int32              => "int".into(),
        Type::Uint32             => "uint".into(),
        Type::Int64               => msl_unsupported_width("int64", "int64_t"),
        Type::Uint64               => msl_unsupported_width("uint64", "uint64_t"),
        Type::Int128               => msl_unsupported_width("int128", "int64_t"),
        Type::Uint128               => msl_unsupported_width("uint128", "uint64_t"),
        Type::Float32            => "float".into(),
        // MSL has no native `double` — a `float64` (or plain `float`, its alias) kernel
        // field used to be silently computed at 32-bit precision on-device with nothing
        // telling the author (docs/float-width-types.md §6). Flagged the same way the
        // unsupported integer widths above already are, instead of a silent narrowing lie.
        Type::Float64            => msl_unsupported_f64("float"),
        Type::Bool             => "bool".into(),
        Type::Str              => "const char*".into(),
        Type::Nil | Type::Void => "void".into(),
        Type::Array(inner)     => format!("{}*", msl_type(inner)),
        Type::ArrayN(inner, n) => format!("{}[{}]", msl_type(inner), n),
        // Const-generic-sized fixed array (`[float, W * H]'const`) — no literal
        // length available here (see `rust_type`'s identical arm in host.rs),
        // so fall back to a pointer like plain `Array` above. Every real 'const
        // array field takes this backend's direct-pointer MSL param path
        // (`constant T* name [[buffer(N)]]`, see this file's `Type::ArrayNExpr`
        // arms above) rather than this fallback.
        Type::ArrayNExpr(inner, _) => format!("{}*", msl_type(inner)),
        Type::LabeledArray(inner, _) => format!("{}*", msl_type(inner)),
        Type::Named(n) => match n.as_str() {
            "float32" | "f32"       => "float".to_string(),
            "float" | "float64" | "f64" => msl_unsupported_f64("float"),
            "int"                   => "int64_t".to_string(),
            "uint"                  => "uint64_t".to_string(),
            "uint8"                 => "uchar".to_string(),
            "int8"                  => "char".to_string(),
            "int16"                 => "short".to_string(),
            "uint16"                => "ushort".to_string(),
            "int32"                 => "int".to_string(),
            "uint32"                => "uint".to_string(),
            "int64" | "i64"         => msl_unsupported_width("int64", "int64_t"),
            "uint64" | "u64"        => msl_unsupported_width("uint64", "uint64_t"),
            "int128" | "i128"       => msl_unsupported_width("int128", "int64_t"),
            "uint128" | "u128"      => msl_unsupported_width("uint128", "uint64_t"),
            "bool"                  => "bool".to_string(),
            other                   => other.to_string(),
        },
        Type::Qualified(inner, _) => msl_type(inner),
        Type::Optional(inner)  => msl_type(inner),
        Type::TypeParam(p)     => p.clone(),
        _                      => "void*".into(),
    }
}

fn elem_msl_type(ty: &Type) -> String {
    match ty {
        Type::Array(inner)        => msl_type(inner),
        Type::ArrayN(inner, _)    => msl_type(inner),
        Type::ArrayNExpr(inner, _) => msl_type(inner),
        Type::Qualified(inner, _) => elem_msl_type(inner),
        Type::LabeledArray(inner, _) => msl_type(inner),
        _                         => msl_type(ty),
    }
}

/// MSL parameter type for a device function's own parameter list -- either a free
/// function (`emit_free_device_fn`) or a kernel struct's own helper method's *extra*
/// (non-field) parameters (`emit_device_fn`). Neither is a kernel STRUCT FIELD (those
/// go through `buffer_field_params`, which already assigns the right address space per
/// `GpuQual`); both instead reach this backend as ordinary Boring function/method
/// parameters, so a `[T]'global`/`'unified`/etc.-qualified array parameter arrives as
/// a plain `Type::Qualified(Type::Array(T), OwnerQual::Gpu*)` — and `msl_type` above
/// unconditionally discards the qualifier for every OTHER use of `Type::Qualified`
/// (borrows, `'actor`, `'shared`, ...), which is correct there but silently drops the
/// address-space requirement here. MSL requires every pointer parameter to declare one
/// (`device`, `constant`, `threadgroup`, `thread`) — omitting it doesn't fail `cargo
/// build` (the Rust host code has no idea), it fails only when the embedded MSL source
/// is compiled by the Metal API at process *runtime* (`new_library_with_source`), with
/// an opaque MSL parse error nowhere near the Boring source. Mirrors the address space
/// `buffer_field_params` already assigns to a kernel struct field of the same qualifier
/// -- including its `const` handling: a kernel field only gets `const` from a `let`
/// binding (`FieldBinding::Let`), never from the qualifier alone, and a plain (no
/// `mut`/`var`) function parameter is the exact same "not writable through here"
/// signal (`Param::mutable` is false) -- omitting it, as an earlier version of this fix
/// did, compiles the MSL cleanly right up until a caller passes a field the kernel
/// itself declared `let` (already `device const T*`, per `buffer_field_params`): MSL
/// then rejects the call for discarding a `const` qualifier, a second runtime-only
/// shader-compile error with the same invisible-to-`cargo-build` failure mode as the
/// missing address space itself.
fn msl_free_fn_param_type(ty: &Type, name: &str, mutable: bool) -> String {
    if let Type::Qualified(inner, oq) = ty {
        let is_array = matches!(inner.as_ref(), Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _))
            || inner.as_labeled_array().is_some();
        if is_array {
            let space = match oq {
                OwnerQual::GpuUnified | OwnerQual::GpuGlobal
                | OwnerQual::GpuActorGlobal | OwnerQual::GpuActorUnified
                | OwnerQual::GpuSurface => Some("device"),
                OwnerQual::GpuConst => Some("constant"),
                // Bare 'local has no host-passed-pointer form in kernel-field position
                // (it's declared as an inline per-thread array in the kernel body, see
                // this file's "Declare 'local arrays" pass) -- but a free function is
                // ordinary Boring syntax, so nothing stops a Boring author writing
                // `[T]'local` as a parameter type. `thread` is MSL's address space for
                // a plain per-thread/stack pointer, matching that same per-thread intent.
                OwnerQual::GpuLocal => Some("thread"),
                _ => None,
            };
            if let Some(space) = space {
                let elem = elem_msl_type(inner);
                // `constant` is already read-only by construction in MSL; the explicit
                // `const` only matters (and is only needed) for `device`/`thread`.
                let constness = if !mutable && space != "constant" { "const " } else { "" };
                return format!("{} {}{}* {}", space, constness, elem, name);
            }
        }
    }
    format!("{} {}", msl_type(ty), name)
}

fn binop_msl(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",  BinOp::Sub => "-",  BinOp::Mul => "*",
        BinOp::Div => "/",  BinOp::Rem => "%",
        BinOp::Eq  => "==", BinOp::NotEq => "!=",
        BinOp::Lt  => "<",  BinOp::Gt => ">",
        BinOp::LtEq => "<=", BinOp::GtEq => ">=",
        BinOp::And => "&&", BinOp::Or => "||",
        BinOp::BitAnd => "&", BinOp::BitOr => "|", BinOp::BitXor => "^",
        BinOp::Shl => "<<", BinOp::Shr => ">>",
        _ => "/*op*/",
    }
}

fn unaryop_msl(op: &UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg    => "-",
        UnaryOp::Not    => "!",
        UnaryOp::BitNot => "~",
    }
}

fn map_gpu_field(obj: &str, field: &str) -> String {
    // gpu.thread / gpu.block etc. map to the kernel built-in parameters.
    match (obj, field) {
        ("gpu", "thread")    => "__thread_pos".into(),
        ("gpu", "block")     => "__block_pos".into(),
        ("gpu", "block_dim" | "blockDim") => "__block_dim".into(),
        ("gpu", "grid_dim" | "gridDim")  => "__grid_dim".into(),
        // Dimension accessors — cast to int64_t for arithmetic.
        ("__thread_pos", "x") => "(int64_t)__thread_pos.x".into(),
        ("__thread_pos", "y") => "(int64_t)__thread_pos.y".into(),
        ("__thread_pos", "z") => "(int64_t)__thread_pos.z".into(),
        ("__block_pos",  "x") => "(int64_t)__block_pos.x".into(),
        ("__block_pos",  "y") => "(int64_t)__block_pos.y".into(),
        ("__block_pos",  "z") => "(int64_t)__block_pos.z".into(),
        ("__block_dim",  "x") => "(int64_t)__block_dim.x".into(),
        ("__block_dim",  "y") => "(int64_t)__block_dim.y".into(),
        ("__block_dim",  "z") => "(int64_t)__block_dim.z".into(),
        ("__grid_dim",   "x") => "(int64_t)__grid_dim.x".into(),
        ("__grid_dim",   "y") => "(int64_t)__grid_dim.y".into(),
        ("__grid_dim",   "z") => "(int64_t)__grid_dim.z".into(),
        ("gpu", "warp")       => "__warp".into(),
        ("__warp", "size")    => "(int64_t)__simd_size".into(),
        ("__warp", "lane")    => "(int64_t)__simd_lane_id".into(),
        _                    => format!("{}.{}", obj, field),
    }
}

/// `gpu.warp.sync()` / `gpu.warp.shuffle_down/up/xor/shuffle(...)` — MSL
/// SIMD-group intrinsics need no capability/mask handling (unlike CUDA's
/// `_sync` mask), so these map straight across.
///
/// One real gap: Metal's `simd_shuffle*` template family is constrained by
/// `__is_valid_simdgroup_type<T>`, which excludes 64-bit integers — only
/// 32-bit-or-narrower scalar types (plus `float`/`bool`) are valid. Boring's
/// default `int`/`uint` map to `int64_t`/`uint64_t` on this backend (see
/// `msl_type`), so `simd_shuffle(v, lane)` on a plain `int` value compiles fine
/// through `boring build` but is rejected by Metal's own compiler at first
/// dispatch (`newLibraryWithSource`) with "no matching function for call to
/// 'simd_shuffle'" — invisible until the kernel actually runs. When
/// `operand_ty` resolves to `int64_t`/`uint64_t`, shuffle the value through a
/// narrower `int32_t`/`uint32_t` view instead and cast back — safe for any
/// value that actually fits in 32 bits, true of every real GPU-side integer
/// use case in this domain. `operand_ty` is `None` when the operand's type
/// couldn't be resolved (see `infer_shuffle_operand_type`); the call is then
/// left unmodified, the same as before this cast existed.
fn gpu_warp_method_call(method: &str, args: &[String], operand_ty: Option<&Type>) -> Option<String> {
    let intrinsic = match method {
        "sync" => return Some("simdgroup_barrier(mem_flags::mem_none)".into()),
        "shuffle_down" | "shuffleDown" => "simd_shuffle_down",
        "shuffle_up" | "shuffleUp"     => "simd_shuffle_up",
        "shuffle_xor" | "shuffleXor"   => "simd_shuffle_xor",
        "shuffle"                     => "simd_shuffle",
        _ => return None,
    };
    let wide_ty = operand_ty.map(msl_type).filter(|t| t == "int64_t" || t == "uint64_t");
    Some(match wide_ty {
        Some(wide) => {
            let narrow = if wide == "int64_t" { "int32_t" } else { "uint32_t" };
            format!("({wide})({intrinsic}(({narrow})({}), {}))", args[0], args[1])
        }
        None => format!("{intrinsic}({}, {})", args[0], args[1]),
    })
}

#[test]
fn tensor_scalar_fallback_is_emitted() {
    let program = crate::transpiler::tensor::test_program();
    let source = emit_device_msl(&program);
    assert!(source.contains("bp_tensor_sum += a["), "{source}");
    assert!(source.contains("c["), "{source}");
    assert!(!source.contains("matmulTile("), "{source}");
}

fn is_gpu_warp_receiver(obj: &Expr) -> bool {
    matches!(&obj.kind, ExprKind::Field(inner, name) if name == "warp"
        && matches!(&inner.kind, ExprKind::Var(v) if v == "gpu"))
}

fn map_builtin_fn(name: &str) -> String {
    match name {
        "int"   => "(int64_t)".into(),
        "float" => "(float)".into(),
        // `float32(x)` — MSL's only float type already is 32-bit, so this is the
        // exact same cast as bare `float(x)` just above. Was missing entirely
        // (fell through to the `other => other.into()` passthrough below, emitting
        // an invalid, undeclared `float32(x)` function call in the generated MSL —
        // confirmed via examples/mandelbrot_gpu.br, which must use `float32`
        // throughout its kernel: MSL has no native `double`, so any GPU kernel
        // field/local touching float data needs `float32`, not the bare
        // `float`/`float64` alias — see docs/float-width-types.md and
        // examples/saxpy.br's identical requirement).
        "float32" => "(float)".into(),
        "abs"   => "abs".into(),
        "min"   => "min".into(),
        "max"   => "max".into(),
        "sqrt"  => "sqrt".into(),
        "exp"   => "exp".into(),
        "log"   => "log".into(),
        "log2"  => "log2".into(),
        "log10" => "log10".into(),
        "sin"   => "sin".into(),
        "cos"   => "cos".into(),
        "tan"   => "tan".into(),
        "tanh"  => "tanh".into(),
        "pow"   => "pow".into(),
        "floor" => "floor".into(),
        "ceil"  => "ceil".into(),
        "round" => "round".into(),
        other   => other.into(),
    }
}

/// Maps Boring's built-in float methods (`x.exp()`, `x.sqrt()`, ...) to MSL
/// `metal_stdlib` calls. Mirrors the Rust-target mapping in
/// `transpiler::emit_methods`'s `FLOAT_UNARY_METHODS` (same method set, MSL
/// names instead of Rust `f64` method names). Returns `None` for methods this
/// device emitter doesn't recognize, so the caller can fall back to its
/// `/* unsupported */` marker.
fn float_unary_method_msl(method: &str, obj: &str, args: &[String]) -> Option<String> {
    let simple = match method {
        "sqrt" => "sqrt", "cbrt" => "cbrt", "abs" => "fabs",
        "floor" => "floor", "ceil" => "ceil", "round" => "round",
        "exp" => "exp", "exp2" => "exp2", "ln" => "log",
        "log2" => "log2", "log10" => "log10",
        "sin" => "sin", "cos" => "cos", "tan" => "tan",
        "asin" => "asin", "acos" => "acos", "atan" => "atan",
        "sinh" => "sinh", "cosh" => "cosh", "tanh" => "tanh",
        _ => "",
    };
    if !simple.is_empty() {
        return Some(format!("{}({})", simple, obj));
    }
    match method {
        "pow" | "powf" => {
            let exp = args.first().cloned().unwrap_or_else(|| "1.0".into());
            Some(format!("pow({}, {})", obj, exp))
        }
        "log" => {
            let base = args.first().cloned().unwrap_or_else(|| "M_E_F".into());
            Some(format!("(log({}) / log({}))", obj, base))
        }
        "atan2" => {
            let other = args.first().cloned().unwrap_or_else(|| "0.0".into());
            Some(format!("atan2({}, {})", obj, other))
        }
        "signum" => Some(format!("sign({})", obj)),
        "recip"  => Some(format!("(1.0 / {})", obj)),
        "toRadians" => Some(format!("({} * (M_PI_F / 180.0))", obj)),
        "toDegrees" => Some(format!("({} * (180.0 / M_PI_F))", obj)),
        _ => None,
    }
}

// ── Auto-sync helpers ─────────────────────────────────────────────────────────

/// Returns true if the body contains at least one explicit `sync` statement at any depth.
fn body_has_explicit_sync(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Comment(c) if c == "sync" => true,
        Stmt::While(w)   => body_has_explicit_sync(&w.body),
        Stmt::For(f)     => body_has_explicit_sync(&f.body),
        Stmt::If(i)      => i.branches.iter().any(|(_, b)| body_has_explicit_sync(b))
                         || i.else_body.as_ref().is_some_and(|b| body_has_explicit_sync(b)),
        _ => false,
    })
}

/// Returns true if the body references any field declared with `'sync`.
fn body_accesses_sync_field(stmts: &[Stmt], fields: &[KernelFieldDecl]) -> bool {
    let sync_names: Vec<&str> = fields.iter()
        .filter(|f| matches!(f.qual, GpuQual::Actor))
        .map(|f| f.name.as_str())
        .collect();
    if sync_names.is_empty() { return false; }
    stmts_reference_any(stmts, &sync_names)
}

fn stmts_reference_any(stmts: &[Stmt], names: &[&str]) -> bool {
    stmts.iter().any(|s| stmt_references_any(s, names))
}

fn stmt_references_any(stmt: &Stmt, names: &[&str]) -> bool {
    match stmt {
        Stmt::Expr(e) => expr_references_any(e, names),
        Stmt::Return(r) => r.value.as_ref().is_some_and(|v| expr_references_any(v, names)),
        Stmt::Let(s) => s.value.as_ref().is_some_and(|v| expr_references_any(v, names)),
        Stmt::While(w) => stmts_reference_any(&w.body, names),
        Stmt::For(f)   => stmts_reference_any(&f.body, names),
        Stmt::If(i)    => i.branches.iter().any(|(_, b)| stmts_reference_any(b, names))
                       || i.else_body.as_ref().is_some_and(|b| stmts_reference_any(b, names)),
        _ => false,
    }
}

fn expr_references_any(expr: &Expr, names: &[&str]) -> bool {
    match &expr.kind {
        ExprKind::Var(n)           => names.contains(&n.as_str()),
        ExprKind::Index(a, i)      => expr_references_any(a, names) || expr_references_any(i, names),
        ExprKind::Field(e, _)      => expr_references_any(e, names),
        ExprKind::BinOp(_, l, r)   => expr_references_any(l, names) || expr_references_any(r, names),
        ExprKind::UnaryOp(_, e)    => expr_references_any(e, names),
        ExprKind::Assign(l, r)     => expr_references_any(l, names) || expr_references_any(r, names),
        ExprKind::Call(f, args)    => expr_references_any(f, names) || args.iter().any(|a| expr_references_any(&a.value, names)),
        _ => false,
    }
}
