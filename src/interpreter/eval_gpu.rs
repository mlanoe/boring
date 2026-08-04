// GPU kernel interpreter — simulation mode.
//
// In simulation mode (`boring run` or default), GPU kernels execute on the CPU.
// Each kernel thread runs independently with its own interpreter instance,
// allowing rayon to parallelize across threads while the main interpreter
// stays single-threaded (Rc-based).
//
// Parallelism strategy:
//   1. Snapshot kernel field values and the captured env into `ThreadValue`
//      (a Send+Sync mirror of `Value` that contains no Rc).
//   2. Use rayon to run each thread's entry-point body concurrently,
//      each with a fresh `Interpreter::new_for_kernel`.
//   3. Merge per-thread results back: scalars take the last write;
//      arrays do element-wise merge (GPU contract: non-overlapping writes).
//   4. Wrap the merged kernel object in a `KernelHandle`.

use super::*;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Barrier, Mutex, OnceLock};
use rayon::prelude::*;
use crate::ast::{GpuQual, TraitDecl, EnumDecl, FnDecl, Stmt, Type};

// Per-thread view of `'sync` kernel fields plus the barrier used to coordinate
// reads/writes across threads within a block (see `run_one_kernel_thread`'s doc
// comment for why `'sync` fields need this instead of the usual snapshot/merge).
pub(crate) type SyncFieldsMap = HashMap<String, Arc<Mutex<Vec<ThreadValue>>>>;
type SyncCtx<'a> = (&'a SyncFieldsMap, &'a Arc<Barrier>);
// Per-thread warp coordination context: this thread's count of active lanes,
// plus the barrier/scratch shared across the warp for `gpu.warp.*` primitives.
type WarpCtx<'a> = (usize, &'a Arc<Barrier>, &'a Arc<Mutex<Vec<ThreadValue>>>);

// Fixed warp/wavefront/SIMD-group/subgroup size for the CPU simulation. Real
// hardware varies (32 on NVIDIA and RDNA AMD, 64 on CDNA AMD, adapter-reported
// on wgpu) — 32 is the common case and matches the same fallback constant
// already used for the unrelated host-side `GPU().warpSize()` introspection
// mock (`src/transpiler/wgpu/host.rs`'s `__boring_gpu_warp_size`).
pub(crate) const WARP_SIZE: usize = 32;

// One warp-group's coordination state within a block: the barrier every
// participating lane waits on, and the shared scratch slots `gpu.warp.shuffle_*`
// reads/writes through (always `WARP_SIZE` slots regardless of how many lanes
// actually participate — a block smaller than `WARP_SIZE` just leaves the
// unused tail unused, same as real hardware's inactive lanes in a partial warp).
pub(crate) type WarpGroup = (Arc<Barrier>, Arc<Mutex<Vec<ThreadValue>>>);

/// Does this expression contain a `gpu.warp.*` field access or method call
/// anywhere in its tree? Used to decide whether a kernel dispatch needs the
/// real-OS-thread, barrier-synchronized execution path (see
/// `run_kernel_parallel`) even when the kernel declares no `'actor` field —
/// `gpu.warp.sync()`/`shuffle_*()` need genuine cross-thread coordination
/// just like a `'actor` field's barrier does, so a kernel that only uses
/// `gpu.warp.*` (no `'actor` fields at all) must still take that path rather
/// than the independent-threads fast path.
fn is_gpu_warp_receiver(e: &Expr) -> bool {
    matches!(&e.kind, ExprKind::Field(inner, name) if name == "warp"
        && matches!(&inner.kind, ExprKind::Var(v) if v == "gpu"))
}

fn expr_uses_gpu_warp(e: &Expr) -> bool {
    if is_gpu_warp_receiver(e) {
        return true;
    }
    match &e.kind {
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Str(_) | ExprKind::Bool(_)
        | ExprKind::Nil | ExprKind::Void | ExprKind::Var(_) | ExprKind::DotIdent(_) => false,
        ExprKind::StringInterp(segs) => segs.iter().any(|s| match s {
            StringSegment::Lit(_) => false,
            StringSegment::Expr(e) => expr_uses_gpu_warp(e),
            StringSegment::FormattedExpr(e, _) => expr_uses_gpu_warp(e),
        }),
        ExprKind::BinOp(_, l, r) => expr_uses_gpu_warp(l) || expr_uses_gpu_warp(r),
        ExprKind::UnaryOp(_, x) => expr_uses_gpu_warp(x),
        ExprKind::Assign(l, r) | ExprKind::QuestionAssign(l, r) =>
            expr_uses_gpu_warp(l) || expr_uses_gpu_warp(r),
        ExprKind::Field(obj, _) | ExprKind::OptionalField(obj, _) => expr_uses_gpu_warp(obj),
        ExprKind::Index(a, i) => expr_uses_gpu_warp(a) || expr_uses_gpu_warp(i),
        ExprKind::LabeledIndex(a, args) =>
            expr_uses_gpu_warp(a) || args.iter().any(|arg| expr_uses_gpu_warp(&arg.value)),
        ExprKind::Call(callee, args) =>
            expr_uses_gpu_warp(callee) || args.iter().any(|a| expr_uses_gpu_warp(&a.value)),
        ExprKind::MethodCall(obj, _, args) | ExprKind::OptionalMethodCall(obj, _, args) =>
            expr_uses_gpu_warp(obj) || args.iter().any(|a| expr_uses_gpu_warp(&a.value)),
        ExprKind::GenericCall(callee, _, args) =>
            expr_uses_gpu_warp(callee) || args.iter().any(|a| expr_uses_gpu_warp(&a.value)),
        ExprKind::Pipe(lhs, _, args) =>
            expr_uses_gpu_warp(lhs) || args.iter().any(|a| expr_uses_gpu_warp(&a.value)),
        ExprKind::New { arena, ctor } =>
            arena.as_ref().map(|a| expr_uses_gpu_warp(a)).unwrap_or(false) || expr_uses_gpu_warp(ctor),
        ExprKind::KernelLaunch { config, kernel } =>
            expr_uses_gpu_warp(kernel)
                || config.block.as_ref().map(expr_uses_gpu_warp).unwrap_or(false)
                || config.grid.as_ref().map(expr_uses_gpu_warp).unwrap_or(false)
                || config.after.as_ref().map(expr_uses_gpu_warp).unwrap_or(false),
        ExprKind::TryElse(a, b) => expr_uses_gpu_warp(a) || expr_uses_gpu_warp(b),
        ExprKind::TryElseBlock(body, else_body) =>
            stmts_use_gpu_warp(body) || stmts_use_gpu_warp(else_body),
        ExprKind::Array(items) | ExprKind::Tuple(items) | ExprKind::Set(items) =>
            items.iter().any(expr_uses_gpu_warp),
        ExprKind::ArrayFill { value, count } =>
            expr_uses_gpu_warp(value) || expr_uses_gpu_warp(count),
        ExprKind::ArrayAlloc { count } => expr_uses_gpu_warp(count),
        ExprKind::ArrayComp { expr, count, .. } =>
            expr_uses_gpu_warp(expr) || expr_uses_gpu_warp(count),
        ExprKind::ArrayCompIter { expr, iter, .. } =>
            expr_uses_gpu_warp(expr) || expr_uses_gpu_warp(iter),
        ExprKind::LabeledArrayComp { expr, clauses } =>
            expr_uses_gpu_warp(expr) || clauses.iter().any(|(_, count)| expr_uses_gpu_warp(count)),
        ExprKind::RelabelCast(x, _) => expr_uses_gpu_warp(x),
        ExprKind::Dict(pairs) =>
            pairs.iter().any(|(k, v)| expr_uses_gpu_warp(k) || expr_uses_gpu_warp(v)),
        ExprKind::Range { start, end, .. } => expr_uses_gpu_warp(start) || expr_uses_gpu_warp(end),
        ExprKind::SliceRange { start, end, .. } =>
            start.as_ref().map(|s| expr_uses_gpu_warp(s)).unwrap_or(false)
                || end.as_ref().map(|e| expr_uses_gpu_warp(e)).unwrap_or(false),
        ExprKind::Cast(x, _) => expr_uses_gpu_warp(x),
        ExprKind::Else(a, b) => expr_uses_gpu_warp(a) || expr_uses_gpu_warp(b),
        ExprKind::Closure(_, _, body, _, _) => match body {
            ClosureBody::Expr(e) => expr_uses_gpu_warp(e),
            ClosureBody::Block(stmts) => stmts_use_gpu_warp(stmts),
        },
        ExprKind::If(s) => if_stmt_uses_gpu_warp(s),
        ExprKind::Match(s) => match_stmt_uses_gpu_warp(s),
        ExprKind::Block(stmts) | ExprKind::Do(stmts) => stmts_use_gpu_warp(stmts),
        ExprKind::Loop(s) => stmts_use_gpu_warp(&s.body),
        ExprKind::Task(x) => expr_uses_gpu_warp(x),
        ExprKind::TaskWithTimeout(a, b) => expr_uses_gpu_warp(a) || expr_uses_gpu_warp(b),
        ExprKind::JoinAll(items) => items.iter().any(expr_uses_gpu_warp),
        ExprKind::MacroCall { args, .. } => args.iter().any(expr_uses_gpu_warp),
    }
}

fn if_stmt_uses_gpu_warp(s: &crate::ast::IfStmt) -> bool {
    s.branches.iter().any(|(c, b)| expr_uses_gpu_warp(c) || stmts_use_gpu_warp(b))
        || s.else_body.as_ref().map(|b| stmts_use_gpu_warp(b)).unwrap_or(false)
}

fn match_stmt_uses_gpu_warp(s: &crate::ast::MatchStmt) -> bool {
    expr_uses_gpu_warp(&s.subject) || s.arms.iter().any(|a| {
        a.guard.as_ref().map(expr_uses_gpu_warp).unwrap_or(false) || match &a.body {
            MatchBody::Expr(e) => expr_uses_gpu_warp(e),
            MatchBody::Block(b) => stmts_use_gpu_warp(b),
        }
    })
}

fn cond_clauses_use_gpu_warp(clauses: &[CondClause]) -> bool {
    clauses.iter().any(|c| match c {
        CondClause::Let(_, e) => expr_uses_gpu_warp(e),
        CondClause::LetPat(_, e) => expr_uses_gpu_warp(e),
        CondClause::Expr(e) => expr_uses_gpu_warp(e),
    })
}

/// Does any statement in `stmts` contain a `gpu.warp.*` field access or method
/// call anywhere in its tree? Shared with the wgpu transpiler backend (see
/// `crate::transpiler::wgpu::kernel_uses_gpu_warp`), which needs the same
/// detection to decide whether to emit the `enable subgroups;` WGSL directive
/// and the SIMD-group builtin kernel parameters for a given kernel.
pub(crate) fn stmts_use_gpu_warp(stmts: &[Stmt]) -> bool {
    stmts.iter().any(stmt_uses_gpu_warp)
}

fn stmt_uses_gpu_warp(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Let(s) => s.value.as_ref().map(expr_uses_gpu_warp).unwrap_or(false),
        Stmt::LetDestructure(s) => expr_uses_gpu_warp(&s.value),
        Stmt::Return(s) => s.value.as_ref().map(expr_uses_gpu_warp).unwrap_or(false),
        Stmt::Break(_, v) => v.as_ref().map(expr_uses_gpu_warp).unwrap_or(false),
        Stmt::Continue(_) => false,
        Stmt::Throw(s) => s.value.as_ref().map(expr_uses_gpu_warp).unwrap_or(false),
        Stmt::If(s) => if_stmt_uses_gpu_warp(s),
        Stmt::IfLet(s) =>
            cond_clauses_use_gpu_warp(&s.clauses) || stmts_use_gpu_warp(&s.then_body)
                || s.elif_branches.iter().any(|b| cond_clauses_use_gpu_warp(&b.clauses) || stmts_use_gpu_warp(&b.body))
                || s.else_body.as_ref().map(|b| stmts_use_gpu_warp(b)).unwrap_or(false),
        Stmt::Match(s) => match_stmt_uses_gpu_warp(s),
        Stmt::While(s) => expr_uses_gpu_warp(&s.condition) || stmts_use_gpu_warp(&s.body),
        Stmt::WhileLet(s) => expr_uses_gpu_warp(&s.value) || stmts_use_gpu_warp(&s.body),
        Stmt::DoWhile(s) => stmts_use_gpu_warp(&s.body) || expr_uses_gpu_warp(&s.condition),
        Stmt::Loop(s) => stmts_use_gpu_warp(&s.body),
        Stmt::Wait(e, _) => expr_uses_gpu_warp(e),
        Stmt::For(s) => expr_uses_gpu_warp(&s.iterable) || stmts_use_gpu_warp(&s.body),
        Stmt::Guard(s) => {
            let cond_uses = match &s.cond {
                GuardCond::Expr(e) => expr_uses_gpu_warp(e),
                GuardCond::Clauses(cs) => cond_clauses_use_gpu_warp(cs),
            };
            cond_uses || stmts_use_gpu_warp(&s.else_body)
        }
        Stmt::Try(s) => stmts_use_gpu_warp(&s.body) || s.catch_clauses.iter().any(|c| stmts_use_gpu_warp(&c.body)),
        Stmt::Defer(body) => stmts_use_gpu_warp(body),
        Stmt::Expr(e) => expr_uses_gpu_warp(e),
        Stmt::Fn(f) => stmts_use_gpu_warp(&f.body),
        Stmt::Struct(_) | Stmt::Enum(_) | Stmt::Mod(_) | Stmt::Alias(_) => false,
        Stmt::Yield(e, _) => expr_uses_gpu_warp(e),
        Stmt::Comment(_) => false,
        Stmt::KernelBlock(s) => stmts_use_gpu_warp(&s.body),
        Stmt::With(s) => stmts_use_gpu_warp(&s.body),
    }
}

// Thread pool with an enlarged stack (64 MB) for the recursive tree-walk interpreter.
static KERNEL_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

fn kernel_pool() -> &'static rayon::ThreadPool {
    KERNEL_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .stack_size(64 * 1024 * 1024)
            .build()
            .expect("failed to build kernel thread pool")
    })
}

// ─── ThreadValue ─────────────────────────────────────────────────────────────
// A Send+Sync version of Value for passing data across kernel threads.
// Covers every type that can appear as a kernel field or env binding.
// Closures and channels are omitted (not valid kernel data).

#[derive(Clone)]
pub(crate) enum ThreadValue {
    Nil,
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
    Array(Vec<ThreadValue>),
    Object { type_name: String, fields: Vec<(String, ThreadValue)> },
    EnumVariant { type_name: String, variant: String, fields: Vec<ThreadValue> },
    Fn(Box<crate::ast::FnDecl>),
}

impl PartialEq for ThreadValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ThreadValue::Nil,    ThreadValue::Nil)    => true,
            (ThreadValue::Void,   ThreadValue::Void)   => true,
            (ThreadValue::Bool(a),  ThreadValue::Bool(b))  => a == b,
            (ThreadValue::Int(a),   ThreadValue::Int(b))   => a == b,
            (ThreadValue::Uint(a),  ThreadValue::Uint(b))  => a == b,
            (ThreadValue::Uint8(a), ThreadValue::Uint8(b)) => a == b,
            (ThreadValue::Int8(a), ThreadValue::Int8(b)) => a == b,
            (ThreadValue::Int16(a), ThreadValue::Int16(b)) => a == b,
            (ThreadValue::Int32(a), ThreadValue::Int32(b)) => a == b,
            (ThreadValue::Int64(a), ThreadValue::Int64(b)) => a == b,
            (ThreadValue::Int128(a), ThreadValue::Int128(b)) => a == b,
            (ThreadValue::Uint16(a), ThreadValue::Uint16(b)) => a == b,
            (ThreadValue::Uint32(a), ThreadValue::Uint32(b)) => a == b,
            (ThreadValue::Uint64(a), ThreadValue::Uint64(b)) => a == b,
            (ThreadValue::Uint128(a), ThreadValue::Uint128(b)) => a == b,
            (ThreadValue::Float(a), ThreadValue::Float(b)) => a == b,
            (ThreadValue::Str(a),   ThreadValue::Str(b))   => a == b,
            (ThreadValue::Array(a), ThreadValue::Array(b)) => a == b,
            (ThreadValue::Object { type_name: tn1, fields: f1 },
             ThreadValue::Object { type_name: tn2, fields: f2 }) => tn1 == tn2 && f1 == f2,
            (ThreadValue::EnumVariant { type_name: t1, variant: v1, fields: f1 },
             ThreadValue::EnumVariant { type_name: t2, variant: v2, fields: f2 }) =>
                t1 == t2 && v1 == v2 && f1 == f2,
            // Fn values are never kernel field data — treat as unequal so writes are detected.
            (ThreadValue::Fn(_), ThreadValue::Fn(_)) => false,
            _ => false,
        }
    }
}

pub(crate) fn to_thread_value(v: &Value) -> Option<ThreadValue> {
    match v {
        Value::Nil                => Some(ThreadValue::Nil),
        Value::Void               => Some(ThreadValue::Void),
        Value::Bool(b)            => Some(ThreadValue::Bool(*b)),
        Value::Int(n)             => Some(ThreadValue::Int(*n)),
        Value::Uint(n)            => Some(ThreadValue::Uint(*n)),
        Value::Uint8(n)           => Some(ThreadValue::Uint8(*n)),
        Value::Int8(n)            => Some(ThreadValue::Int8(*n)),
        Value::Int16(n)           => Some(ThreadValue::Int16(*n)),
        Value::Int32(n)           => Some(ThreadValue::Int32(*n)),
        Value::Int64(n)           => Some(ThreadValue::Int64(*n)),
        Value::Int128(n)          => Some(ThreadValue::Int128(*n)),
        Value::Uint16(n)          => Some(ThreadValue::Uint16(*n)),
        Value::Uint32(n)          => Some(ThreadValue::Uint32(*n)),
        Value::Uint64(n)          => Some(ThreadValue::Uint64(*n)),
        Value::Uint128(n)         => Some(ThreadValue::Uint128(*n)),
        Value::Float(f)           => Some(ThreadValue::Float(*f)),
        Value::Str(s)             => Some(ThreadValue::Str(s.clone())),
        Value::Fn { decl, .. }    => Some(ThreadValue::Fn(Box::new(decl.clone()))),
        Value::Array(arr)         => {
            let tvs: Option<Vec<_>> = arr.iter().map(to_thread_value).collect();
            tvs.map(ThreadValue::Array)
        }
        Value::Object(inner)      => {
            let inner = inner.borrow();
            let fields: Option<Vec<_>> = inner.fields.iter()
                .map(|(k, v)| to_thread_value(v).map(|tv| (k.clone(), tv)))
                .collect();
            fields.map(|f| ThreadValue::Object { type_name: inner.type_name.clone(), fields: f })
        }
        Value::EnumVariant { type_name, variant, fields } => {
            let tvs: Option<Vec<_>> = fields.iter().map(to_thread_value).collect();
            tvs.map(|f| ThreadValue::EnumVariant {
                type_name: type_name.clone(),
                variant: variant.clone(),
                fields: f,
            })
        }
        _ => None,
    }
}

pub(crate) fn from_thread_value(v: ThreadValue, captured: &EnvRef) -> Value {
    match v {
        ThreadValue::Nil              => Value::Nil,
        ThreadValue::Void             => Value::Void,
        ThreadValue::Bool(b)          => Value::Bool(b),
        ThreadValue::Int(n)           => Value::Int(n),
        ThreadValue::Uint(n)          => Value::Uint(n),
        ThreadValue::Uint8(n)         => Value::Uint8(n),
        ThreadValue::Int8(n)          => Value::Int8(n),
        ThreadValue::Int16(n)         => Value::Int16(n),
        ThreadValue::Int32(n)         => Value::Int32(n),
        ThreadValue::Int64(n)         => Value::Int64(n),
        ThreadValue::Int128(n)        => Value::Int128(n),
        ThreadValue::Uint16(n)        => Value::Uint16(n),
        ThreadValue::Uint32(n)        => Value::Uint32(n),
        ThreadValue::Uint64(n)        => Value::Uint64(n),
        ThreadValue::Uint128(n)       => Value::Uint128(n),
        ThreadValue::Float(f)         => Value::Float(f),
        ThreadValue::Str(s)           => Value::Str(s),
        ThreadValue::Fn(decl)         => Value::Fn { decl: *decl, captured: Rc::clone(captured) },
        ThreadValue::Array(arr)       => Value::Array(
            arr.into_iter().map(|tv| from_thread_value(tv, captured)).collect::<Vec<_>>().into()
        ),
        ThreadValue::Object { type_name, fields } => {
            let fields = fields.into_iter()
                .map(|(k, tv)| (k, from_thread_value(tv, captured)))
                .collect();
            Value::Object(Rc::new(RefCell::new(ObjectInner { type_name, fields })))
        }
        ThreadValue::EnumVariant { type_name, variant, fields } => Value::EnumVariant {
            type_name,
            variant,
            fields: fields.into_iter().map(|tv| from_thread_value(tv, captured)).collect(),
        },
    }
}

// ─── Snapshot helpers ─────────────────────────────────────────────────────────

/// Walk the env chain and snapshot every binding that converts to ThreadValue.
/// Stdlib functions are skipped (NativeFn is not convertible; the fresh interpreter
/// already provides them).
fn snapshot_env(env: &EnvRef) -> Vec<(String, ThreadValue)> {
    let mut out = Vec::new();
    let borrowed = env.borrow();
    for (name, val) in &borrowed.vars {
        if let Some(tv) = to_thread_value(val) {
            out.push((name.clone(), tv));
        }
    }
    if let Some(parent) = &borrowed.parent {
        // Walk parent chain but don't re-export stdlib (parent of global is None).
        out.extend(snapshot_env(parent));
    }
    out
}

// ─── Per-thread result ────────────────────────────────────────────────────────

struct ThreadResult {
    /// Changed mutable fields: (field_name, initial_value, new_value).
    /// Stored as ThreadValue so it's Send.
    fields: Vec<(String, ThreadValue, ThreadValue)>,
}

// ─── Core kernel simulation ───────────────────────────────────────────────────

/// Thread/block/grid dimensions for a single kernel launch (`run_kernel_parallel`).
pub(crate) struct LaunchDims {
    pub total_threads: usize,
    pub block_x:       usize,
    pub block_y:       usize,
    pub block_z:       usize,
    pub grid_x:        usize,
    pub grid_y:        usize,
    pub grid_z:        usize,
}

/// Run the kernel's anonymous entry point for `dims.total_threads` threads in parallel.
///
/// Returns the final kernel object with all thread writes merged.
fn run_kernel_parallel(
    interp:     &Interpreter,
    decl:       &crate::ast::KernelDecl,
    captured:   &EnvRef,
    kernel_obj: Value,
    entry:      &crate::ast::FnDecl,
    dims:       LaunchDims,
) -> Result<Value, Signal> {
    let LaunchDims { total_threads, block_x, block_y, block_z, grid_x, grid_y, grid_z } = dims;
    // Snapshot the initial field values and the captured env once (before threads).
    let initial_fields: Vec<(String, ThreadValue)> = if let Value::Object(ref obj) = kernel_obj {
        obj.borrow().fields.iter()
            .map(|(name, val)| (name.clone(), to_thread_value(val).unwrap_or(ThreadValue::Nil)))
            .collect()
    } else {
        vec![]
    };

    let mutable_names: Vec<String> = decl.fields.iter()
        .filter(|f| matches!(f.binding, FieldBinding::Mut | FieldBinding::Var))
        .map(|f| f.name.clone())
        .collect();
    let initial_fields = Arc::new(initial_fields);
    let mutable_names   = Arc::new(mutable_names);

    // Arc-wrapped so the `'sync` path's job closures (run on pooled worker
    // threads that outlive this call) can hold cheap, owned, `'static` clones
    // instead of borrowing from this stack frame. The fast path below just
    // derefs through these the same as it would a plain `&T` — no behavior
    // change there.
    let captured_snapshot = Arc::new(snapshot_env(captured));
    let entry_body        = Arc::new(entry.body.clone());
    let decl_fields       = Arc::new(decl.fields.clone());
    let decl_methods      = Arc::new(decl.methods.clone());
    let traits            = Arc::new(interp.traits.clone());
    let enums_map         = Arc::new(interp.enums.clone());
    let aliases           = Arc::new(interp.aliases.clone());
    let gpu_profile       = Arc::new(interp.gpu_profile.clone());

    // `'sync` fields need real cross-thread visibility within a block (one thread
    // reading another's write after a barrier) — see `run_one_kernel_thread`'s doc
    // comment for why that needs a genuinely different execution strategy than the
    // rest of this function's "run every thread independently, merge afterward" model.
    let sync_field_specs: Vec<(String, crate::ast::Type, usize)> = decl_fields.iter()
        .filter_map(|f| match (&f.qual, &f.ty) {
            (GpuQual::Actor, Type::ArrayN(inner, n)) => Some((f.name.clone(), inner.as_ref().clone(), *n)),
            // An 'actor-qualified `[T, width=16, height=16]` tile field
            // (LinearKernel/MatMulBTHeadsKernel's shared-memory tiles) needs
            // real cross-thread barrier visibility, or each thread silently
            // gets its own independent copy of the tile instead of seeing
            // what its block-mates cooperatively wrote — found via
            // whisper-boring's math_gpu.br migration producing wrong (but not
            // crashing) GEMM results.
            (GpuQual::Actor, ty) if ty.as_labeled_array().is_some() => {
                let (elem, _) = ty.as_labeled_array().unwrap();
                let len = ty.labeled_array_len().expect("checker guarantees fixed-shape literal axes on an 'actor field");
                Some((f.name.clone(), elem.clone(), len as usize))
            }
            _ => None,
        })
        .collect();

    // `gpu.warp.sync()`/`gpu.warp.shuffle_*()` need genuine cross-thread
    // coordination within a block, exactly like a `'actor` field's barrier —
    // so a kernel using them must take the real-OS-thread path below even
    // when it declares no `'actor` field at all.
    let uses_gpu_warp = stmts_use_gpu_warp(&entry.body);

    #[allow(clippy::too_many_arguments)]
    fn run_one_kernel_thread(
        traits: &HashMap<String, TraitDecl>,
        enums_map: &HashMap<String, EnumDecl>,
        aliases: &HashMap<String, Type>,
        gpu_profile: &gpu_profile::GpuProfile,
        captured_snapshot: &[(String, ThreadValue)],
        entry_body: &[Stmt],
        decl_fields: &[crate::ast::KernelFieldDecl],
        decl_methods: &[FnDecl],
        initial_fields: &[(String, ThreadValue)],
        mutable_names: &[String],
        block_x: usize, block_y: usize, block_z: usize,
        grid_x: usize, grid_y: usize, grid_z: usize,
        block_idx_x: usize, block_idx_y: usize, block_idx_z: usize,
        thread_in_x: usize, thread_in_y: usize, thread_in_z: usize,
        sync_ctx: Option<SyncCtx>,
        warp_ctx: Option<WarpCtx>,
    ) -> Result<ThreadResult, String> {
        // Build a fresh interpreter for this thread.
        let mut ti = Interpreter::new_for_kernel(
            traits.clone(),
            enums_map.clone(),
            aliases.clone(),
            gpu_profile.clone(),
        );
        if let Some((sync_fields, barrier)) = sync_ctx {
            ti.sync_fields = sync_fields.clone();
            ti.kernel_barrier = Some(Arc::clone(barrier));
        }
        // This thread's lane within its warp — always computed (cheap, pure
        // arithmetic), regardless of whether the kernel actually uses
        // `gpu.warp.*` (see `WARP_SIZE`'s doc comment for the linearization
        // formula, matching the one CUDA/ROCm codegen emits device-side).
        let flat_thread_for_lane = thread_in_x + thread_in_y * block_x + thread_in_z * block_x * block_y;
        ti.warp_lane = flat_thread_for_lane % WARP_SIZE;
        if let Some((active_lanes, barrier, scratch)) = warp_ctx {
            ti.warp_active_lanes = active_lanes;
            ti.warp_barrier = Some(Arc::clone(barrier));
            ti.warp_scratch = Some(Arc::clone(scratch));
        }

        // Reconstruct the captured env as a child of the fresh global.
        let cap_env = Env::child(Rc::clone(&ti.global));
        for (name, tv) in captured_snapshot {
            let val = from_thread_value(tv.clone(), &cap_env);
            cap_env.borrow_mut().define(name, val);
        }

        // Build the thread env.
        let thread_env = Env::child(Rc::clone(&cap_env));

        // Inject field values (mutable fields are define_mut so the body can write them).
        // `'sync` fields are still bound here too (harmless placeholder — reads/writes
        // to their name are intercepted before ever consulting this binding, see
        // `Interpreter::sync_fields`'s doc comment), so non-kernel code paths that
        // check "is this name bound" keep working unchanged.
        for (field_decl, (name, tv)) in decl_fields.iter().zip(initial_fields.iter()) {
            let val = from_thread_value(tv.clone(), &thread_env);
            match field_decl.binding {
                FieldBinding::Mut | FieldBinding::Var =>
                    thread_env.borrow_mut().define_mut(name, val),
                FieldBinding::Let =>
                    thread_env.borrow_mut().define(name, val),
            }
        }

        // Reconstruct the kernel object as `self`.
        let self_fields: Vec<(String, Value)> = initial_fields.iter()
            .map(|(n, tv)| (n.clone(), from_thread_value(tv.clone(), &thread_env)))
            .collect();
        let self_obj = Value::Object(Rc::new(RefCell::new(ObjectInner {
            type_name: decl_fields.first().map(|_| "".to_string()).unwrap_or_default(),
            fields: self_fields,
        })));
        thread_env.borrow_mut().define("self", self_obj);

        // Inject gpu.* builtins.
        let gpu_thread = make_object("GpuThread".into(), vec![
            ("x".into(), Value::Int(thread_in_x as i64)),
            ("y".into(), Value::Int(thread_in_y as i64)),
            ("z".into(), Value::Int(thread_in_z as i64)),
        ]);
        let gpu_block = make_object("GpuBlock".into(), vec![
            ("x".into(), Value::Int(block_idx_x as i64)),
            ("y".into(), Value::Int(block_idx_y as i64)),
            ("z".into(), Value::Int(block_idx_z as i64)),
        ]);
        let gpu_block_dim = make_object("GpuBlockDim".into(), vec![
            ("x".into(), Value::Int(block_x as i64)),
            ("y".into(), Value::Int(block_y as i64)),
            ("z".into(), Value::Int(block_z as i64)),
        ]);
        let gpu_grid_dim = make_object("GpuGridDim".into(), vec![
            ("x".into(), Value::Int(grid_x as i64)),
            ("y".into(), Value::Int(grid_y as i64)),
            ("z".into(), Value::Int(grid_z as i64)),
        ]);
        let gpu_warp = make_object("GpuWarp".into(), vec![
            ("size".into(), Value::Int(WARP_SIZE as i64)),
            ("lane".into(), Value::Int(ti.warp_lane as i64)),
        ]);
        thread_env.borrow_mut().define("gpu", make_object("Gpu".into(), vec![
            ("thread".into(),    gpu_thread),
            ("block".into(),     gpu_block),
            ("block_dim".into(), gpu_block_dim),
            ("grid_dim".into(),  gpu_grid_dim),
            ("warp".into(),      gpu_warp),
        ]));

        // Inject kernel methods.
        for method in decl_methods {
            if !method.name.is_empty() {
                let fn_val = Value::Fn { decl: method.clone(), captured: Rc::clone(&thread_env) };
                thread_env.borrow_mut().define(&method.name, fn_val);
            }
        }

        // Run the entry point.
        let result = ti.exec_block(entry_body, Rc::clone(&thread_env));
        match result {
            Ok(_) | Err(Signal::Return(_)) => {}
            Err(Signal::Error(e)) => return Err(e.message),
            Err(e) => return Err(format!("{:?}", e)),
        }

        // Collect changed mutable fields (never includes `'sync` fields — those
        // aren't in `mutable_names`'s source list in the caller for kernels with
        // sync_ctx... actually they are FieldBinding::Mut too, so this DOES walk
        // them, but their `thread_env` binding was never updated by the body (all
        // real reads/writes went through `sync_ctx` instead), so it never differs
        // from its initial snapshot and never shows up as "changed" here — matching
        // `'sync` fields never escaping the kernel body, same as real hardware.
        let mut changed = Vec::new();
        for name in mutable_names {
            let initial = initial_fields.iter()
                .find(|(n, _)| n == name)
                .map(|(_, tv)| tv.clone())
                .unwrap_or(ThreadValue::Nil);
            if let Some(new_val) = thread_env.borrow().vars.get(name).and_then(to_thread_value) {
                if new_val != initial {
                    changed.push((name.clone(), initial, new_val));
                }
            }
        }
        Ok(ThreadResult { fields: changed })
    }

    let thread_results: Vec<Result<ThreadResult, String>> = if sync_field_specs.is_empty() && !uses_gpu_warp {
        // Fast path — no `'sync` fields, every thread is independent. Unchanged
        // from before except block/thread z-indices are now real (previously
        // hardcoded to 0/1); existing kernels never dispatch with block_z/grid_z
        // > 1, so block_idx_z/thread_in_z are still always 0 for them — identical
        // behavior, this only newly matters for kernels that use grid.z/block.z.
        kernel_pool().install(|| {
            (0..total_threads).into_par_iter().map(|thread_idx| {
                let threads_per_block = block_x * block_y * block_z;
                let blocks_per_layer   = grid_x * grid_y;
                let flat_block         = thread_idx / threads_per_block;
                let flat_thread        = thread_idx % threads_per_block;
                let block_idx_x        = flat_block % grid_x;
                let block_idx_y        = (flat_block / grid_x) % grid_y;
                let block_idx_z        = flat_block / blocks_per_layer;
                let thread_in_x        = flat_thread % block_x;
                let thread_in_y        = (flat_thread / block_x) % block_y;
                let thread_in_z        = flat_thread / (block_x * block_y);

                run_one_kernel_thread(
                    &traits, &enums_map, &aliases, &gpu_profile,
                    &captured_snapshot, &entry_body, &decl_fields, &decl_methods,
                    &initial_fields, &mutable_names,
                    block_x, block_y, block_z, grid_x, grid_y, grid_z,
                    block_idx_x, block_idx_y, block_idx_z,
                    thread_in_x, thread_in_y, thread_in_z,
                    None,
                    None,
                )
            })
            .collect::<Vec<_>>()
        })
    } else {
        // `'sync` fields are block-scoped shared memory on real hardware: every
        // thread in the same block must observe writes another thread in that
        // block made before the last barrier. That's incompatible with the fast
        // path's model (every thread runs independently to completion, merged
        // afterward) — no thread would ever see another's write. Rayon's
        // work-stealing pool is also unsafe to block threads-within-a-block on a
        // real `Barrier` over: the pool has a fixed worker count, and if a block
        // has more threads than there are free workers, the workers that did get
        // scheduled can deadlock waiting at the barrier for sibling threads that
        // the pool has no free worker left to even start.
        //
        // So: parallelize across BLOCKS with rayon (blocks are always
        // independent, real hardware never synchronizes across them either), and
        // within each block spawn genuine OS threads via `std::thread::scope` —
        // sized exactly to that block's thread count, so every one of them is
        // actually running concurrently and a shared `Barrier` can't deadlock.
        // Real OS threads also sidestep needing to rearchitect this tree-walking
        // interpreter into a suspend/resume (coroutine) model to support a
        // `sync` statement arbitrarily deep inside nested control flow (e.g.
        // inside a `while` loop, as gpu-module.md's manual-mode example does) —
        // blocking a real thread on `Barrier::wait()` just naturally pauses its
        // Rust call stack wherever it happens to be.
        //
        // A prior version of this path tried a persistent, checked-out/in
        // worker-thread pool (parked on a channel between jobs) specifically to
        // avoid the spawn/destroy cost below. Measured, not assumed: it was
        // SLOWER in every variant tried (~65s and ~52s on a test file this
        // version runs in ~32s) — the channel round-trip to wake a parked
        // worker and collect its result apparently costs more than the thread
        // spawn it was meant to save, at this workload's scale (~100 dispatches
        // of up to a few hundred threads each). Spawn-and-destroy per block, as
        // below, measured faster — don't reintroduce a pool here without a new
        // measurement showing it actually wins.
        let threads_per_block = block_x * block_y * block_z;
        let n_blocks = grid_x * grid_y * grid_z;
        let blocks_per_layer = grid_x * grid_y;

        // Plain `&T` references (Copy) to the shared, per-dispatch-constant data —
        // so the innermost `move` closure (spawned once per thread, many times per
        // block, many blocks) copies a reference each time instead of trying to
        // move the same owned `HashMap`/`Vec` out of its enclosing scope repeatedly.
        let traits_r            = traits.as_ref();
        let enums_map_r         = enums_map.as_ref();
        let aliases_r           = aliases.as_ref();
        let gpu_profile_r       = gpu_profile.as_ref();
        let captured_snapshot_r = captured_snapshot.as_ref();
        let entry_body_r        = entry_body.as_ref();
        let decl_fields_r       = decl_fields.as_ref();
        let decl_methods_r      = decl_methods.as_ref();
        let initial_fields_r    = initial_fields.as_ref();
        let mutable_names_r     = mutable_names.as_ref();

        kernel_pool().install(|| {
            (0..n_blocks).into_par_iter().map(|block_idx| {
                let block_idx_x = block_idx % grid_x;
                let block_idx_y = (block_idx / grid_x) % grid_y;
                let block_idx_z = block_idx / blocks_per_layer;

                let sync_fields: SyncFieldsMap = sync_field_specs.iter()
                    .map(|(name, elem_ty, n)| {
                        let zero = to_thread_value(&zero_value(elem_ty)).unwrap_or(ThreadValue::Nil);
                        (name.clone(), Arc::new(Mutex::new(vec![zero; *n])))
                    })
                    .collect();
                let barrier = Arc::new(Barrier::new(threads_per_block.max(1)));

                // One warp-group per `WARP_SIZE`-sized (or smaller, for the
                // block's last partial warp) slice of this block's threads —
                // built unconditionally whenever this (real-OS-thread) path
                // runs, even for kernels that only need it for `'actor`
                // fields, since it's cheap relative to the thread spawns below.
                let n_warp_groups = threads_per_block.div_ceil(WARP_SIZE).max(1);
                let warp_groups: Vec<WarpGroup> = (0..n_warp_groups).map(|w| {
                    let active = threads_per_block.saturating_sub(w * WARP_SIZE).min(WARP_SIZE);
                    (
                        Arc::new(Barrier::new(active.max(1))),
                        Arc::new(Mutex::new(vec![ThreadValue::Nil; WARP_SIZE])),
                    )
                }).collect();

                std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..threads_per_block).map(|flat_thread| {
                        let thread_in_x = flat_thread % block_x;
                        let thread_in_y = (flat_thread / block_x) % block_y;
                        let thread_in_z = flat_thread / (block_x * block_y);
                        let sync_fields = &sync_fields;
                        let barrier = &barrier;
                        let warp_id = flat_thread / WARP_SIZE;
                        let (warp_barrier, warp_scratch) = &warp_groups[warp_id];
                        let warp_active = threads_per_block.saturating_sub(warp_id * WARP_SIZE).min(WARP_SIZE);
                        // Enlarged (but not `kernel_pool()`-sized) stack — this
                        // tree-walking interpreter's recursion can blow the
                        // OS-default stack size (see that pool's own doc
                        // comment), and a stack overflow deep inside a thread
                        // that's mid-recursion while holding a `'sync` field's
                        // Mutex (or waiting at the Barrier) hangs every other
                        // thread in the block rather than failing loudly. 8 MB
                        // (not 64 MB) — verified against the full test suite
                        // (486+ interpreter tests, including real nested-loop
                        // kernel bodies) with no overflow; smaller reservations
                        // measurably speed up repeated spawn/destroy cycles.
                        std::thread::Builder::new()
                            .stack_size(64 * 1024 * 1024)
                            .spawn_scoped(scope, move || {
                                run_one_kernel_thread(
                                    traits_r, enums_map_r, aliases_r, gpu_profile_r,
                                    captured_snapshot_r, entry_body_r, decl_fields_r, decl_methods_r,
                                    initial_fields_r, mutable_names_r,
                                    block_x, block_y, block_z, grid_x, grid_y, grid_z,
                                    block_idx_x, block_idx_y, block_idx_z,
                                    thread_in_x, thread_in_y, thread_in_z,
                                    Some((sync_fields, barrier)),
                                    Some((warp_active, warp_barrier, warp_scratch)),
                                )
                            })
                            .expect("failed to spawn kernel simulation thread")
                    }).collect();
                    handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("kernel thread panicked".to_string()))).collect::<Vec<_>>()
                })
            })
            .flatten()
            .collect::<Vec<_>>()
        })
    }; // thread_results

    // Check for any thread errors.
    let mut all_results = Vec::with_capacity(total_threads);
    for r in thread_results {
        match r {
            Ok(tr) => all_results.push(tr),
            Err(msg) => return Err(err(msg, 0)),
        }
    }

    // Merge results back into kernel_obj using element-wise merge for arrays.
    if let Value::Object(ref obj) = kernel_obj {
        for name in mutable_names.iter() {
            let base_tv = initial_fields.iter()
                .find(|(n, _)| n == name)
                .map(|(_, tv)| tv.clone())
                .unwrap_or(ThreadValue::Nil);

            let merged_tv = match &base_tv {
                ThreadValue::Array(base_arr) => {
                    // Element-wise merge: for each element, take the first thread write that
                    // differs from the base value. GPU contract guarantees non-overlapping
                    // writes, so at most one thread changes each index.
                    let mut merged = base_arr.clone();
                    for tr in &all_results {
                        for (f_name, f_init, f_new) in &tr.fields {
                            if f_name != name { continue; }
                            if let (ThreadValue::Array(init_arr), ThreadValue::Array(new_arr)) =
                                (f_init, f_new)
                            {
                                for (i, (init_elem, new_elem)) in
                                    init_arr.iter().zip(new_arr.iter()).enumerate()
                                {
                                    if new_elem != init_elem {
                                        if let Some(slot) = merged.get_mut(i) {
                                            *slot = new_elem.clone();
                                        }
                                    }
                                }
                            }
                        }
                    }
                    ThreadValue::Array(merged)
                }
                _ => {
                    // Scalar: last write wins (matches sequential semantics).
                    let mut last = base_tv;
                    for tr in &all_results {
                        for (f_name, _, f_new) in &tr.fields {
                            if f_name == name {
                                last = f_new.clone();
                            }
                        }
                    }
                    last
                }
            };

            // Write merged value back into the shared kernel object.
            let val = from_thread_value(merged_tv, &Rc::new(RefCell::new(Env {
                parent: None,
                vars: Default::default(),
                mutable: Default::default(),
                owned_collections: Default::default(),
                task_safe_vars: Default::default(),
                owned_vars: Default::default(),
                shared_bindings: Default::default(),
                actor_bindings: Default::default(),
                lazy_vars: Default::default(),
            })));
            let mut o = obj.borrow_mut();
            if let Some(entry) = o.fields.iter_mut().find(|(n, _)| n == name) {
                entry.1 = val;
            }
        }
    }

    Ok(kernel_obj)
}

// ─── Interpreter impl ─────────────────────────────────────────────────────────

impl Interpreter {
    /// `gpu.warp.sync()` / `gpu.warp.shuffle_down/up/xor/shuffle(...)` — intercepted
    /// syntactically in `eval_expr_method_call` before generic method dispatch (the
    /// receiver `gpu.warp` is never a real bound value, only `gpu.warp.size`/`.lane`
    /// are, via the `GpuWarp` object `run_one_kernel_thread` injects).
    pub(crate) fn eval_gpu_warp_method(&mut self, method: &str, args: &[Arg], env: EnvRef, line: usize) -> Eval {
        match method {
            "sync" => {
                if let Some(barrier) = &self.warp_barrier {
                    barrier.wait();
                }
                Ok(Value::Void)
            }
            "shuffle_down" | "shuffle_up" | "shuffle_xor" | "shuffle" => {
                if args.len() != 2 {
                    return Err(err(format!("gpu.warp.{} expects 2 arguments", method), line));
                }
                let v = self.eval_expr(&args[0].value, Rc::clone(&env))?;
                let other = self.eval_expr(&args[1].value, Rc::clone(&env))?;
                let operand = match other {
                    Value::Int(n) => n,
                    Value::Uint(n) => n as i64,
                    _ => return Err(err(
                        format!("gpu.warp.{}'s second argument must be an integer", method), line,
                    )),
                };
                let lane = self.warp_lane as i64;
                let target_lane: i64 = match method {
                    "shuffle_down" => lane + operand,
                    "shuffle_up"   => lane - operand,
                    "shuffle_xor"  => lane ^ operand,
                    "shuffle"      => operand,
                    _ => unreachable!(),
                };
                let (barrier, scratch) = match (&self.warp_barrier, &self.warp_scratch) {
                    (Some(b), Some(s)) => (Arc::clone(b), Arc::clone(s)),
                    // Not running under a warp-synchronized kernel dispatch — shouldn't
                    // happen (`stmts_use_gpu_warp` forces that path for any kernel that
                    // reaches this call), but degrade to "read your own value" rather
                    // than panicking if it ever does.
                    _ => return Ok(v),
                };
                let tv = to_thread_value(&v).ok_or_else(|| err(
                    format!("gpu.warp.{}'s value argument isn't a shareable kernel type", method), line,
                ))?;
                {
                    let mut slots = scratch.lock().unwrap();
                    slots[self.warp_lane] = tv;
                }
                barrier.wait();
                let result_tv = {
                    let slots = scratch.lock().unwrap();
                    if target_lane >= 0 && (target_lane as usize) < self.warp_active_lanes {
                        slots[target_lane as usize].clone()
                    } else {
                        // Out-of-range lane: matches real hardware's `_sync` shuffle
                        // intrinsics returning the caller's own value at the warp's
                        // boundary rather than reading a nonexistent/inactive lane.
                        slots[self.warp_lane].clone()
                    }
                };
                barrier.wait();
                Ok(from_thread_value(result_tv, &env))
            }
            other => Err(err(format!("gpu.warp has no method '{}'", other), line)),
        }
    }

    /// Register a `kernel Name:` declaration in the environment.
    pub(crate) fn exec_kernel_decl(&mut self, decl: &crate::ast::KernelDecl, env: EnvRef) -> Result<(), Signal> {
        let decl = lower_labeled_array_methods(decl);
        let decl = &decl;
        let val = Value::KernelStruct { decl: decl.clone(), captured: Rc::clone(&env) };
        env.borrow_mut().define(&decl.name, val.clone());
        if !Rc::ptr_eq(&env, &self.global) {
            self.global.borrow_mut().define(&decl.name, val);
        }
        Ok(())
    }

    /// Evaluate a `kernel(block = N, ...) expr` expression.
    pub(crate) fn eval_kernel_launch(
        &mut self,
        config: &crate::ast::KernelConfig,
        kernel_expr: &Expr,
        env: EnvRef,
    ) -> Eval {
        let kernel_val = self.eval_expr(kernel_expr, Rc::clone(&env))?;

        let (type_name, fields, decl, captured) = match &kernel_val {
            Value::Object(inner) => {
                let type_name = inner.borrow().type_name.clone();
                let fields    = inner.borrow().fields.clone();
                let kval      = env.borrow().get(&type_name);
                match kval {
                    Some(Value::KernelStruct { decl, captured }) => (type_name, fields, decl, captured),
                    _ => return Ok(Value::KernelHandle { result: Box::new(kernel_val) }),
                }
            }
            _ => return Ok(Value::KernelHandle { result: Box::new(kernel_val) }),
        };

        let (block_x, block_y, block_z) = if let Some(block_expr) = &config.block {
            match self.eval_expr(block_expr, Rc::clone(&env))? {
                Value::Int(n) => (n.max(1) as usize, 1, 1),
                Value::Tuple(t) => {
                    let x = match t.first() { Some(Value::Int(n)) => (*n).max(1) as usize, _ => 1 };
                    let y = match t.get(1)  { Some(Value::Int(n)) => (*n).max(1) as usize, _ => 1 };
                    let z = match t.get(2)  { Some(Value::Int(n)) => (*n).max(1) as usize, _ => 1 };
                    (x, y, z)
                }
                _ => (1, 1, 1),
            }
        } else {
            (1, 1, 1)
        };

        let (grid_x, grid_y, grid_z) = if let Some(grid_expr) = &config.grid {
            match self.eval_expr(grid_expr, Rc::clone(&env))? {
                Value::Int(n) => (n.max(1) as usize, 1, 1),
                Value::Tuple(t) => {
                    let x = match t.first() { Some(Value::Int(n)) => (*n).max(1) as usize, _ => 1 };
                    let y = match t.get(1)  { Some(Value::Int(n)) => (*n).max(1) as usize, _ => 1 };
                    let z = match t.get(2)  { Some(Value::Int(n)) => (*n).max(1) as usize, _ => 1 };
                    (x, y, z)
                }
                _ => (1, 1, 1),
            }
        } else {
            let max_len = fields.iter()
                .filter_map(|(_, v)| if let Value::Array(a) = v { Some(a.len()) } else { None })
                .max()
                .unwrap_or(0);
            let inferred_x = if max_len > 0 { max_len.div_ceil(block_x * block_y * block_z) } else { 1 };
            (inferred_x, 1, 1)
        };

        let total_threads = block_x * block_y * block_z * grid_x * grid_y * grid_z;

        let entry = decl.methods.iter().find(|m| m.name.is_empty() && m.params.is_empty());
        if let Some(entry) = entry {
            let kernel_obj = make_object(type_name, fields);
            let kernel_obj = run_kernel_parallel(
                self, &decl, &captured, kernel_obj, entry,
                LaunchDims { total_threads, block_x, block_y, block_z, grid_x, grid_y, grid_z },
            )?;
            Ok(Value::KernelHandle { result: Box::new(kernel_obj) })
        } else {
            Ok(Value::KernelHandle { result: Box::new(make_object(type_name, fields)) })
        }
    }

    /// `k(block = N)` short-hand dispatch.
    pub(crate) fn eval_kernel_launch_with_val(
        &mut self,
        _config: crate::ast::KernelConfig,
        kernel_val: Value,
        block_val: Value,
        _line: usize,
        env: &EnvRef,
    ) -> Eval {
        let (type_name, fields, decl, captured) = match &kernel_val {
            Value::Object(inner) => {
                let type_name = inner.borrow().type_name.clone();
                let fields    = inner.borrow().fields.clone();
                let kval      = env.borrow().get(&type_name);
                match kval {
                    Some(Value::KernelStruct { decl, captured }) => (type_name, fields, decl, captured),
                    _ => return Ok(Value::KernelHandle { result: Box::new(kernel_val) }),
                }
            }
            _ => return Ok(Value::KernelHandle { result: Box::new(kernel_val) }),
        };

        let (block_x, block_y, block_z) = match block_val {
            Value::Int(n)   => (n.max(1) as usize, 1, 1),
            Value::Uint(n)  => (n.max(1) as usize, 1, 1),
            Value::Tuple(ref t) => {
                let x = match t.first() {
                    Some(Value::Int(n))  => (*n).max(1) as usize,
                    Some(Value::Uint(n)) => (*n).max(1) as usize,
                    _ => 1,
                };
                let y = match t.get(1) {
                    Some(Value::Int(n))  => (*n).max(1) as usize,
                    Some(Value::Uint(n)) => (*n).max(1) as usize,
                    _ => 1,
                };
                let z = match t.get(2) {
                    Some(Value::Int(n))  => (*n).max(1) as usize,
                    Some(Value::Uint(n)) => (*n).max(1) as usize,
                    _ => 1,
                };
                (x, y, z)
            }
            _ => (1, 1, 1),
        };

        let max_len = fields.iter()
            .filter_map(|(_, v)| if let Value::Array(a) = v { Some(a.len()) } else { None })
            .max()
            .unwrap_or(0);

        let (grid_x, grid_y) = if max_len > 0 {
            let bxy = block_x * block_y * block_z;
            let dim_w = fields.iter().find(|(n, _)| n == "dim")
                .and_then(|(_, v)| if let Value::Object(o) = v {
                    o.borrow().fields.iter().find(|(k, _)| k == "width")
                        .and_then(|(_, w)| if let Value::Uint(n) = w { Some(*n as usize) } else { None })
                } else { None });
            let dim_h = fields.iter().find(|(n, _)| n == "dim")
                .and_then(|(_, v)| if let Value::Object(o) = v {
                    o.borrow().fields.iter().find(|(k, _)| k == "height")
                        .and_then(|(_, h)| if let Value::Uint(n) = h { Some(*n as usize) } else { None })
                } else { None });
            if let (Some(w), Some(h)) = (dim_w, dim_h) {
                (w.div_ceil(block_x), h.div_ceil(block_y))
            } else {
                (max_len.div_ceil(bxy), 1)
            }
        } else {
            (1, 1)
        };
        let grid_z = 1;

        let total_threads = block_x * block_y * block_z * grid_x * grid_y * grid_z;

        let entry = decl.methods.iter().find(|m| m.name.is_empty() && m.params.is_empty());
        if let Some(entry) = entry {
            let kernel_obj = make_object(type_name, fields);
            let kernel_obj = run_kernel_parallel(
                self, &decl, &captured, kernel_obj, entry,
                LaunchDims { total_threads, block_x, block_y, block_z, grid_x, grid_y, grid_z },
            )?;
            Ok(Value::KernelHandle { result: Box::new(kernel_obj) })
        } else {
            Ok(Value::KernelHandle { result: Box::new(make_object(type_name, fields)) })
        }
    }

    /// Instantiate a kernel struct: call its `init` declaration (if any).
    pub(crate) fn instantiate_kernel_struct(
        &mut self,
        decl: &crate::ast::KernelDecl,
        captured: &EnvRef,
        args: Vec<Value>,
        _line: usize,
    ) -> Eval {
        let mut fields: Vec<(String, Value)> = decl.fields
            .iter()
            .map(|f| (f.name.clone(), Value::Nil))
            .collect();

        let init_decl = decl.inits.iter()
            .find(|i| {
                if i.params.len() != args.len() { return false; }
                if let (Some(first_param), Some(first_arg)) = (i.params.first(), args.first()) {
                    let expected = first_param.ty.as_ref().map(|t| match t {
                        crate::ast::Type::Named(n) => n.as_str(),
                        _ => "",
                    });
                    let actual = first_arg.type_name();
                    if let Some(exp) = expected {
                        if !exp.is_empty() && exp != actual.as_str() { return false; }
                    }
                }
                true
            })
            .or_else(|| decl.inits.first());

        if let Some(init_decl) = init_decl {
            let env = Env::child(Rc::clone(captured));
            let mut positional: std::collections::VecDeque<Value> = args.into_iter().collect();
            let param_names: std::collections::HashSet<String> =
                init_decl.params.iter().map(|p| p.name.clone()).collect();
            for param in &init_decl.params {
                let val = positional.pop_front().unwrap_or(Value::Nil);
                env.borrow_mut().define_mut(&param.name, val);
            }
            for (_, (name, val)) in decl.fields.iter().zip(fields.iter()) {
                if !param_names.contains(name) {
                    env.borrow_mut().define_mut(name, val.clone());
                }
            }
            match self.exec_block(&init_decl.body, Rc::clone(&env)) {
                Ok(_) | Err(Signal::Return(_)) => {}
                Err(e) => return Err(e),
            }
            for (name, val) in fields.iter_mut() {
                if let Some(new_val) = env.borrow().get(name) {
                    *val = new_val;
                }
            }
        } else if !args.is_empty() {
            for (i, arg) in args.into_iter().enumerate() {
                if let Some((_, v)) = fields.get_mut(i) {
                    *v = arg;
                }
            }
        }

        // Fixed-size array fields (`[T, N]'sync` / `[T, N]'local`) don't need an
        // init() assignment — real GPU targets declare them unconditionally
        // (WGSL `var<workgroup>`/CUDA `__shared__` are zero-initialized by the
        // hardware) — see gpu-module.md's "no init() assignment needed". Mirror
        // that here instead of leaving the field `Nil`, which would hard-error
        // on the first `tile[i] = ...` inside the kernel body.
        for (field_decl, (_, val)) in decl.fields.iter().zip(fields.iter_mut()) {
            if matches!(val, Value::Nil) {
                if let crate::ast::Type::ArrayN(inner, n) = &field_decl.ty {
                    *val = Value::Array(vec![zero_value(inner); *n].into());
                } else if let Some((elem, _)) = field_decl.ty.as_labeled_array() {
                    // A fixed-shape LabeledArray field is represented the same
                    // as a flat ArrayN of the same total length — `LabeledIndex`/
                    // `.size(.axis)` have already been lowered to a plain
                    // `Index` into this same flat array (see
                    // `lower_labeled_array_methods`), so no separate shape
                    // tracking is needed at runtime. Only when every axis size
                    // is a literal int — a LabeledArray axis may be an
                    // arbitrary const-generic expression (`width = W`), which
                    // `labeled_array_len()` can't fold without a subst-map
                    // evaluation context this loop doesn't have. Left `Nil` in
                    // that case — not yet exercised by any real kernel (every
                    // real .br fixed-shape tile uses literal sizes).
                    if let Some(len) = field_decl.ty.labeled_array_len() {
                        *val = Value::Array(vec![zero_value(elem); len as usize].into());
                    }
                }
            }
        }

        Ok(make_object(decl.name.clone(), fields))
    }

    /// Execute a `kernel:` block body.
    pub(crate) fn exec_kernel_block(&mut self, stmts: &[crate::ast::Stmt], env: EnvRef) -> Result<(), Signal> {
        let prev = self.kernel_context;
        self.kernel_context = true;
        let result = self.exec_kernel_block_inner(stmts, env);
        self.kernel_context = prev;
        result
    }

    fn exec_kernel_block_inner(&mut self, stmts: &[crate::ast::Stmt], env: EnvRef) -> Result<(), Signal> {
        use crate::ast::Stmt;
        use crate::ast::ExprKind;

        for stmt in stmts {
            match stmt {
                Stmt::Expr(e) => {
                    let val = self.eval_expr(e, Rc::clone(&env))?;
                    if let Value::KernelHandle { result } = val {
                        let var_name = match &e.kind {
                            ExprKind::Call(callee, _) => {
                                if let ExprKind::Var(name) = &callee.kind { Some(name.clone()) } else { None }
                            }
                            _ => None,
                        };
                        if let Some(name) = var_name {
                            env.borrow_mut().force_set(&name, *result);
                        }
                    }
                }
                Stmt::Loop(l) => {
                    match self.exec_loop(l, Rc::clone(&env)) {
                        Ok(_) | Err(Signal::Break(_)) => {}
                        Err(e) => return Err(e),
                    }
                }
                other => {
                    self.exec_stmt(other, Rc::clone(&env))?;
                }
            }
        }
        Ok(())
    }
}

// ─── Labeled multi-dim array lowering (fixed-shape kernel fields only) ─────
//
// Lowers `.at`-equivalent `LabeledIndex`/`.size(.axis)` calls on
// `Type::LabeledArray` kernel fields (docs/array-multidim-types.md). By the
// time this runs, any *dynamic*-shape LabeledArray field has already been
// rewritten away by `desugar_labeled_array` (a whole-program pre-pass, run
// once before the program is ever interpreted) into a plain buffer field +
// shadow `uint` fields — so `as_labeled_array()` here only ever sees a fixed
// shape (every axis a compile-time size). TryElseBlock/Match/Block/Do/
// Closure/New/KernelLaunch/Dict/Set/StringInterp/SliceRange/DotIdent aren't
// realistic inside a kernel body's numeric hot path, so they're left
// unrewritten (falls through to `other => other.clone()`).

pub(crate) fn lower_labeled_array_methods(decl: &crate::ast::KernelDecl) -> crate::ast::KernelDecl {
    let labeled_fields: LabeledArrayFieldAxes = decl.fields.iter()
        .filter_map(|f| f.ty.as_labeled_array().map(|(_, axes)| (f.name.clone(), axes.to_vec())))
        .collect();
    if labeled_fields.is_empty() {
        return decl.clone();
    }
    let mut lowered = decl.clone();
    for method in lowered.methods.iter_mut() {
        method.body = lower_labeled_stmts(&method.body, &labeled_fields);
    }
    for init in lowered.inits.iter_mut() {
        init.body = lower_labeled_stmts(&init.body, &labeled_fields);
    }
    lowered
}

type LabeledArrayFieldAxes = HashMap<String, Vec<crate::ast::LabeledAxis>>;

fn lower_labeled_stmts(stmts: &[Stmt], fields: &LabeledArrayFieldAxes) -> Vec<Stmt> {
    stmts.iter().map(|s| lower_labeled_stmt(s, fields)).collect()
}

fn lower_labeled_stmt(stmt: &Stmt, fields: &LabeledArrayFieldAxes) -> Stmt {
    use crate::ast::MatchBody;
    match stmt {
        Stmt::Let(s) => {
            let mut s = s.clone();
            s.value = s.value.as_ref().map(|v| lower_labeled_expr(v, fields));
            Stmt::Let(s)
        }
        Stmt::LetDestructure(s) => {
            let mut s = s.clone();
            s.value = lower_labeled_expr(&s.value, fields);
            Stmt::LetDestructure(s)
        }
        Stmt::Return(r) => {
            let mut r = r.clone();
            r.value = r.value.as_ref().map(|v| lower_labeled_expr(v, fields));
            Stmt::Return(r)
        }
        Stmt::Break(line, val) => Stmt::Break(*line, val.as_ref().map(|v| lower_labeled_expr(v, fields))),
        Stmt::Throw(t) => {
            let mut t = t.clone();
            t.value = t.value.as_ref().map(|v| lower_labeled_expr(v, fields));
            Stmt::Throw(t)
        }
        Stmt::If(i) => {
            let mut i = i.clone();
            i.branches = i.branches.iter().map(|(c, b)| (lower_labeled_expr(c, fields), lower_labeled_stmts(b, fields))).collect();
            i.else_body = i.else_body.as_ref().map(|b| lower_labeled_stmts(b, fields));
            Stmt::If(i)
        }
        Stmt::IfLet(i) => {
            let mut i = i.clone();
            i.clauses = i.clauses.iter().map(|c| lower_labeled_cond_clause(c, fields)).collect();
            i.then_body = lower_labeled_stmts(&i.then_body, fields);
            i.elif_branches = i.elif_branches.iter().map(|b| {
                let mut b = b.clone();
                b.clauses = b.clauses.iter().map(|c| lower_labeled_cond_clause(c, fields)).collect();
                b.body = lower_labeled_stmts(&b.body, fields);
                b
            }).collect();
            i.else_body = i.else_body.as_ref().map(|b| lower_labeled_stmts(b, fields));
            Stmt::IfLet(i)
        }
        Stmt::Match(m) => {
            let mut m = m.clone();
            m.subject = lower_labeled_expr(&m.subject, fields);
            m.arms = m.arms.iter().map(|arm| {
                let mut arm = arm.clone();
                arm.guard = arm.guard.as_ref().map(|g| lower_labeled_expr(g, fields));
                arm.body = match &arm.body {
                    MatchBody::Expr(e) => MatchBody::Expr(lower_labeled_expr(e, fields)),
                    MatchBody::Block(b) => MatchBody::Block(lower_labeled_stmts(b, fields)),
                };
                arm
            }).collect();
            Stmt::Match(m)
        }
        Stmt::While(w) => {
            let mut w = w.clone();
            w.condition = lower_labeled_expr(&w.condition, fields);
            w.body = lower_labeled_stmts(&w.body, fields);
            Stmt::While(w)
        }
        Stmt::WhileLet(w) => {
            let mut w = w.clone();
            w.body = lower_labeled_stmts(&w.body, fields);
            Stmt::WhileLet(w)
        }
        Stmt::DoWhile(d) => {
            let mut d = d.clone();
            d.body = lower_labeled_stmts(&d.body, fields);
            d.condition = lower_labeled_expr(&d.condition, fields);
            Stmt::DoWhile(d)
        }
        Stmt::Loop(l) => {
            let mut l = l.clone();
            l.body = lower_labeled_stmts(&l.body, fields);
            Stmt::Loop(l)
        }
        Stmt::Wait(e, line) => Stmt::Wait(lower_labeled_expr(e, fields), *line),
        Stmt::For(f) => {
            let mut f = f.clone();
            f.iterable = lower_labeled_expr(&f.iterable, fields);
            f.body = lower_labeled_stmts(&f.body, fields);
            Stmt::For(f)
        }
        Stmt::Guard(g) => {
            let mut g = g.clone();
            g.else_body = lower_labeled_stmts(&g.else_body, fields);
            Stmt::Guard(g)
        }
        Stmt::Try(t) => {
            let mut t = t.clone();
            t.body = lower_labeled_stmts(&t.body, fields);
            t.catch_clauses = t.catch_clauses.iter().map(|c| {
                let mut c = c.clone();
                c.body = lower_labeled_stmts(&c.body, fields);
                c
            }).collect();
            Stmt::Try(t)
        }
        Stmt::Defer(body) => Stmt::Defer(lower_labeled_stmts(body, fields)),
        Stmt::Expr(e) => Stmt::Expr(lower_labeled_expr(e, fields)),
        Stmt::Yield(e, line) => Stmt::Yield(lower_labeled_expr(e, fields), *line),
        other => other.clone(),
    }
}

fn lower_labeled_cond_clause(c: &crate::ast::CondClause, fields: &LabeledArrayFieldAxes) -> crate::ast::CondClause {
    use crate::ast::CondClause;
    match c {
        CondClause::Let(name, e) => CondClause::Let(name.clone(), lower_labeled_expr(e, fields)),
        CondClause::LetPat(p, e) => CondClause::LetPat(p.clone(), lower_labeled_expr(e, fields)),
        CondClause::Expr(e) => CondClause::Expr(lower_labeled_expr(e, fields)),
    }
}

fn lower_labeled_arg(a: &Arg, fields: &LabeledArrayFieldAxes) -> Arg {
    let mut a = a.clone();
    a.value = lower_labeled_expr(&a.value, fields);
    a
}

fn lower_labeled_expr(e: &Expr, fields: &LabeledArrayFieldAxes) -> Expr {
    use crate::ast::ExprKind;
    let kind = match &e.kind {
        ExprKind::LabeledIndex(obj, args) => {
            if let ExprKind::Var(name) = &obj.kind {
                if let Some(axes) = fields.get(name) {
                    if let Some(offset) = labeled_index_offset(args, axes, fields, e.line, e.col) {
                        return Expr {
                            kind: ExprKind::Index(Box::new(lower_labeled_expr(obj, fields)), Box::new(offset)),
                            line: e.line, col: e.col, len: e.len,
                        };
                    }
                }
            }
            ExprKind::LabeledIndex(
                Box::new(lower_labeled_expr(obj, fields)),
                args.iter().map(|a| lower_labeled_arg(a, fields)).collect(),
            )
        }
        ExprKind::MethodCall(obj, method, args) => {
            if method == "size" {
                if let ExprKind::Var(name) = &obj.kind {
                    if let Some(axes) = fields.get(name) {
                        if let [arg] = args.as_slice() {
                            if let ExprKind::DotIdent(axis) = &arg.value.kind {
                                if let Some(resolved) = resolve_labeled_size_call(axes, axis) {
                                    return resolved;
                                }
                            }
                        }
                    }
                }
            }
            ExprKind::MethodCall(Box::new(lower_labeled_expr(obj, fields)), method.clone(),
                args.iter().map(|a| lower_labeled_arg(a, fields)).collect())
        }
        ExprKind::BinOp(op, l, r) => ExprKind::BinOp(op.clone(), Box::new(lower_labeled_expr(l, fields)), Box::new(lower_labeled_expr(r, fields))),
        ExprKind::UnaryOp(op, v) => ExprKind::UnaryOp(op.clone(), Box::new(lower_labeled_expr(v, fields))),
        ExprKind::Assign(l, r) => ExprKind::Assign(Box::new(lower_labeled_expr(l, fields)), Box::new(lower_labeled_expr(r, fields))),
        ExprKind::QuestionAssign(l, r) => ExprKind::QuestionAssign(Box::new(lower_labeled_expr(l, fields)), Box::new(lower_labeled_expr(r, fields))),
        ExprKind::Field(o, name) => ExprKind::Field(Box::new(lower_labeled_expr(o, fields)), name.clone()),
        ExprKind::Index(a, i) => ExprKind::Index(Box::new(lower_labeled_expr(a, fields)), Box::new(lower_labeled_expr(i, fields))),
        ExprKind::Call(callee, args) => ExprKind::Call(Box::new(lower_labeled_expr(callee, fields)), args.iter().map(|a| lower_labeled_arg(a, fields)).collect()),
        ExprKind::GenericCall(callee, tys, args) => ExprKind::GenericCall(Box::new(lower_labeled_expr(callee, fields)), tys.clone(), args.iter().map(|a| lower_labeled_arg(a, fields)).collect()),
        ExprKind::Pipe(l, name, args) => ExprKind::Pipe(Box::new(lower_labeled_expr(l, fields)), name.clone(), args.iter().map(|a| lower_labeled_arg(a, fields)).collect()),
        ExprKind::TryElse(a, b) => ExprKind::TryElse(Box::new(lower_labeled_expr(a, fields)), Box::new(lower_labeled_expr(b, fields))),
        ExprKind::Array(elems) => ExprKind::Array(elems.iter().map(|x| lower_labeled_expr(x, fields)).collect()),
        ExprKind::ArrayFill { value, count } => ExprKind::ArrayFill { value: Box::new(lower_labeled_expr(value, fields)), count: Box::new(lower_labeled_expr(count, fields)) },
        ExprKind::ArrayAlloc { count } => ExprKind::ArrayAlloc { count: Box::new(lower_labeled_expr(count, fields)) },
        ExprKind::ArrayComp { expr, var, count } => ExprKind::ArrayComp { expr: Box::new(lower_labeled_expr(expr, fields)), var: var.clone(), count: Box::new(lower_labeled_expr(count, fields)) },
        ExprKind::ArrayCompIter { expr, var, iter } => ExprKind::ArrayCompIter { expr: Box::new(lower_labeled_expr(expr, fields)), var: var.clone(), iter: Box::new(lower_labeled_expr(iter, fields)) },
        ExprKind::Tuple(xs) => ExprKind::Tuple(xs.iter().map(|x| lower_labeled_expr(x, fields)).collect()),
        ExprKind::Range { start, end, inclusive } => ExprKind::Range { start: Box::new(lower_labeled_expr(start, fields)), end: Box::new(lower_labeled_expr(end, fields)), inclusive: *inclusive },
        ExprKind::Cast(inner, ty) => ExprKind::Cast(Box::new(lower_labeled_expr(inner, fields)), ty.clone()),
        ExprKind::Else(a, b) => ExprKind::Else(Box::new(lower_labeled_expr(a, fields)), Box::new(lower_labeled_expr(b, fields))),
        ExprKind::OptionalField(o, name) => ExprKind::OptionalField(Box::new(lower_labeled_expr(o, fields)), name.clone()),
        ExprKind::OptionalMethodCall(o, name, args) => ExprKind::OptionalMethodCall(Box::new(lower_labeled_expr(o, fields)), name.clone(), args.iter().map(|a| lower_labeled_arg(a, fields)).collect()),
        ExprKind::If(i) => {
            let mut i = i.clone();
            i.branches = i.branches.iter().map(|(c, b)| (lower_labeled_expr(c, fields), lower_labeled_stmts(b, fields))).collect();
            i.else_body = i.else_body.as_ref().map(|b| lower_labeled_stmts(b, fields));
            ExprKind::If(i)
        }
        // TryElseBlock/Match/Block/Do/Closure/New/KernelLaunch/Dict/Set/
        // StringInterp/SliceRange/DotIdent — not realistic inside a kernel
        // body's numeric hot path (see this section's top comment).
        other => other.clone(),
    };
    Expr { kind, line: e.line, col: e.col, len: e.len }
}

/// Row-major flat offset for `args` (a `LabeledIndex`'s labeled arguments)
/// into `axes` — every axis's size is a compile-time `ConstExpr` here (only
/// fixed-shape LabeledArray fields reach this function; see this section's
/// top comment), spliced into the offset expression as-is rather than
/// folded to a literal, since it may reference a kernel const-generic param
/// (`width = W`) resolved normally when the offset is later evaluated, not
/// by this lowering pass. `None` if any label in `args` doesn't match one of
/// `axes` — left unresolved rather than guessed.
fn labeled_index_offset(args: &[Arg], axes: &[crate::ast::LabeledAxis], fields: &LabeledArrayFieldAxes, line: usize, col: usize) -> Option<Expr> {
    use crate::ast::{ExprKind, BinOp, ConstExpr};
    let axis_size_expr = |i: usize| -> Expr {
        let ConstExpr(boxed) = axes[i].size.as_ref()
            .expect("fixed-shape kernel field: every axis has Some(size) by construction");
        (**boxed).clone()
    };
    let mut flat: Option<Expr> = None;
    for (i, axis) in axes.iter().enumerate() {
        let arg = args.iter().find(|a| a.label.as_deref() == Some(axis.label.as_str()))?;
        let idx_expr = lower_labeled_expr(&arg.value, fields);
        let term = if i == 0 {
            idx_expr
        } else {
            let stride = (0..i).map(axis_size_expr)
                .reduce(|acc, v| Expr { kind: ExprKind::BinOp(BinOp::Mul, Box::new(acc), Box::new(v)), line, col, len: 0 })
                .expect("i >= 1 implies a non-empty stride range");
            Expr { kind: ExprKind::BinOp(BinOp::Mul, Box::new(idx_expr), Box::new(stride)), line, col, len: 0 }
        };
        flat = Some(match flat {
            None => term,
            Some(prev) => Expr { kind: ExprKind::BinOp(BinOp::Add, Box::new(prev), Box::new(term)), line, col, len: 0 },
        });
    }
    flat
}

/// `.size(.axis)`'s resolved value for a fixed-shape field — its axis's
/// literal/const-generic-expression size, cloned as-is. `None` if
/// `axis_label` doesn't match any of `axes`.
fn resolve_labeled_size_call(axes: &[crate::ast::LabeledAxis], axis_label: &str) -> Option<Expr> {
    use crate::ast::ConstExpr;
    let axis = axes.iter().find(|a| a.label == axis_label)?;
    let ConstExpr(boxed) = axis.size.as_ref()?;
    Some((**boxed).clone())
}

/// Zero value for a kernel field's scalar element type — used to default
/// `[T, N]'sync`/`[T, N]'local` fields the kernel's `init()` never assigns
/// (see `instantiate_kernel_struct`).
fn zero_value(ty: &crate::ast::Type) -> Value {
    use crate::ast::Type;
    match ty {
        Type::Float => Value::Float(0.0),
        Type::Bool => Value::Bool(false),
        Type::Uint => Value::Uint(0),
        Type::Uint8 => Value::Uint8(0),
        Type::Uint16 => Value::Uint16(0),
        Type::Uint32 => Value::Uint32(0),
        Type::Uint64 => Value::Uint64(0),
        Type::Uint128 => Value::Uint128(0),
        Type::Int8 => Value::Int8(0),
        Type::Int16 => Value::Int16(0),
        Type::Int32 => Value::Int32(0),
        Type::Int64 => Value::Int64(0),
        Type::Int128 => Value::Int128(0),
        _ => Value::Int(0),
    }
}

use crate::ast::{FieldBinding, Expr};
