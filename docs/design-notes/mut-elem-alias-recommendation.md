# `let q = arr[i]` aliasing: recommendation

**Rule chosen: a file-level form of idea 2, gated by idea 1's trigger.** If a file binds an element read with `let`/`mut`/`var` (`let q = coll[i]`), every declared `[mut T]` / `{K = mut V}` type in it with a same-file non-generic struct `T` becomes `'actor` (`src/infer_actor_elems.rs`, a parse-time rewrite like `infer_actor_params`, so checker/interpreter/transpiler agree).

Why not per-collection detection (idea 1 proper): the element type is *declared* in fields, parameters and returns, and a call crossing a function boundary has to agree on the Rust type (`Vec<T>` vs `Vec<Arc<Mutex<T>>>`). Deciding per variable from local usage cannot keep those consistent; a declaration-level rewrite can.
Why not the blanket idea 2: every `[mut T]` in hot loops would pay a lock for no benefit when no alias is ever taken. The trigger (an element-read binding exists in the file) keeps alias-free programs on plain `Vec<T>`.

Trade-offs / what breaks
- A lock per element access in files that alias; `[mut T'inline]` opts out.
- Cross-file: a `[mut T]` in another file is not rewritten (the trigger and `T` are per file); passing across that boundary would then be a rustc type mismatch. Same-file use is the supported case.
- Skipped: structs a kernel mentions (or field types of one), generic structs, foreign structs, plain `[T]`, and everything under `--linux`.
- Array/dict literals are wrapped only when initializing a declared `let`; struct-field defaults / assignments of a literal are not wrapped yet.
- Tested: simd/wgpu build and run (alias prints 9). On metal/cuda/rocm a file with a kernel goes through a separate host emitter that ignores qualifiers and writes `arr[i].clone().x = v` (a write to a temporary): `arr[0].x = 9.0` is silently lost there, with or without this rule (pre-existing bug, now fixed: `emit_assign_target` handles an index base, so the write is in place). The three host emitters now also honour `'actor` elements (`Arc<Mutex<T>>` types, wrapped array/dict literals and `push`, `let q = coll[i]` as a handle clone, locked field read/write and method calls through `lock()`): metal runs and matches the interpreter; cuda/rocm are checked as generated text only. Fresh values stored by `arr[i] = T(..)` / `d[k] = T(..)` are wrapped too. Struct-field collections work too (`b.cells[0].x = v`, `b.cells.push(T(..))`, `let s = b.cells[0]`). cuda/rocm only checked as generated text (no toolchain).
- `--linux`: `[T'actor]` emits `Vec<Arc<Mutex<T>>>` but without literal wrapping or locked element access, so the pass is disabled there.
