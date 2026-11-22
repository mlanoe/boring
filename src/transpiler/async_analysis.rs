// Whole-program, fixed-point "does this function need to be `async fn`" analysis — split out
// of `mod.rs`, following the same pattern as `gpu_analysis.rs`/`compute_gpu_device_only_fns`.
//
// Before this pass existed, async-ness was decided per function, at EMIT time, purely from
// `self.task_fns` as it stood at that point in the single top-to-bottom pass over the
// program's items (see `emit_top.rs`'s `implicit_async`). A function calling another
// already-known-async function got promoted correctly only when its callee happened to have
// been emitted (and so registered into `task_fns`) earlier in file/`use` order — anything
// else (a callee declared later in the same file, or a multi-hop chain where the leaf's own
// async-ness comes from something deeper) was silently missed, producing a plain `fn` that
// later needs `.await` inside it: `error[E0728]: await is only allowed inside async
// functions`, or — when the callee's own parameter type also flips sync/async on
// `self.in_async` (see `emit_top.rs`'s `emit_type`, `Type::Named` arm) — a same-shaped
// mismatched-type error instead (confirmed via boring-llm's ROCm-target server: `serve_conn`
// never got promoted even though it transitively drives tokio's async `TcpStream` through
// `read_http_request`/`dispatch_request`/`handle_completion`/`respond*`/`write_text`).
//
// `compute_implicit_async_fns` replaces that single forward pass with a real whole-program
// fixed point, run once per file (same granularity as `compute_gpu_arg_params`/
// `compute_gpu_device_only_fns` above) BEFORE any function in that file is emitted, so
// `self.task_fns` is already complete by the time `emit_top.rs`'s own (now redundant, but
// harmless) per-function check runs.

use super::*;
use super::Transpiler;

impl Transpiler {
    /// Computes the transitive closure of "needs to be `async fn`" over every free (non-
    /// method, `qualifier.is_none()`) function declared in `program`'s own top-level items,
    /// and merges the result into `self.task_fns`. Mirrors `implicit_async`'s existing
    /// per-function criteria for the *direct* (non-transitive) part — same seeds, so a
    /// function this pass marks async is exactly one `emit_top.rs`'s own check would
    /// (eventually, order permitting) have marked async too — plus one new direct seed
    /// (`fn_body_has_direct_tokio_async_call`, see its doc) and a proper fixed point for the
    /// transitive part instead of a single forward sweep.
    ///
    /// Scope matches `implicit_async`'s own `self_ty.is_none()` restriction: a method is
    /// never implicitly async (`emit_top.rs` turns a non-`task` method's use of an async
    /// stream into a checker error instead, naming the `task` fix) — this pass leaves that
    /// restriction exactly as it is and only strengthens the free-function case.
    pub(crate) fn compute_implicit_async_fns(&mut self, program: &Program) {
        let mut all_fns: Vec<&FnDecl> = Vec::new();
        for item in &program.items { Self::gather_gpu_arg_fns_item(item, &mut all_fns); }
        let all_fns: Vec<&FnDecl> = all_fns.into_iter().filter(|f| f.qualifier.is_none()).collect();

        let has_any_explicit_async = all_fns.iter().any(|f| {
            f.task
                || body_has_channel_or_task(&f.body)
                || body_has_stream_for(&f.body, &self.stream_fns, &self.stream_async_method_names)
                || fn_has_task_closure_param(f, &self.fn_type_aliases)
        });

        // Seed: every function whose own body needs async regardless of who calls it.
        let mut async_fns: std::collections::HashSet<String> = all_fns.iter()
            .filter(|f| {
                f.task
                    || body_has_channel_or_task(&f.body)
                    || body_has_stream_for(&f.body, &self.stream_fns, &self.stream_async_method_names)
                    || fn_has_task_closure_param(f, &self.fn_type_aliases)
                    || (has_any_explicit_async && Self::fn_body_has_direct_tokio_async_call(f))
            })
            .map(|f| f.name.clone())
            .collect();

        // Fixed point: a function that calls an already-async function needs to be async too
        // (so it can `.await` that call) — propagate until nothing new is found. Finite and
        // monotonic (a function only ever moves from "not async" to "async", never back), so
        // this always terminates, same reasoning as `compute_gpu_device_only_fns` above.
        loop {
            let mut changed = false;
            for f in &all_fns {
                if async_fns.contains(&f.name) { continue; }
                if body_calls_task_fn(&f.body, &async_fns) {
                    async_fns.insert(f.name.clone());
                    changed = true;
                }
            }
            if !changed { break; }
        }

        self.task_fns.extend(async_fns);
    }

    /// True if `f`'s body directly calls a known dual sync/async stdlib operation
    /// (`Transpiler::TOKIO_ASYNC_METHODS`/`TOKIO_ASYNC_STATIC_TYPE`, emit_methods.rs) on a
    /// receiver that is either one of `f`'s own parameters declared with a dual sync/async
    /// stdlib type (`TcpStream`, `TcpListener`, ... — `tokio_async_module_for`), or that dual
    /// type's own name (a static call, e.g. `TcpListener.bind(...)`).
    ///
    /// This is a REAL async requirement independent of `f`'s caller: whichever flavor
    /// (`std`/`tokio`) of the dual type ends up chosen for this parameter at emission time,
    /// the call only type-checks as the tokio flavor's method (`impl Future`, needing
    /// `.await`) if `f` itself is `async fn` — the std flavor returns a plain `Result`
    /// instead. So a function built to run this call at all (the common real-world shape:
    /// a socket handed through as a plain parameter, used directly) needs to be async
    /// regardless of whether anything upstream already forced it to be, exactly like a
    /// direct `task:`/`channel(...)` in the body already does. Used as a seed by
    /// `compute_implicit_async_fns`, which then fixed-points the transitive closure over
    /// every caller of such a function.
    ///
    /// Only resolves a receiver that is a bare parameter reference (not a local re-typed by a
    /// `let`/`var` with its own annotation, nor a field/method-call chain) — the real-world
    /// shape this was built for (a socket parameter used directly) never needs more, and a
    /// narrower, conservative detector here is exactly what a *seed* should be: a false
    /// negative just falls through to the (still correct, just order-dependent) existing
    /// per-function check as a fallback, while a false positive would wrongly force a
    /// function async.
    ///
    /// Restricted to the network dual types (`TOKIO_ASYNC_NETWORK_TYPES`) — NOT `File`, even
    /// though `tokio_async_module_for` resolves it too. `File`'s own disk I/O is a vastly more
    /// common shape than a connection socket to find behind a plain parameter with no
    /// surrounding `task:`/`channel` content at all (reading/writing a snapshot file is
    /// ordinary, deliberately-synchronous code in plenty of programs, confirmed via boring-
    /// llm's own `prompt_cache.br`: `save_snapshot_file`'s `File.create(path)` is called from
    /// `ServerState::commit`, a plain **method** — which can never become async itself, see
    /// `emit_top.rs`'s "a method is never implicitly async" — so seeding `save_snapshot_file`
    /// here would force it async with no way for its caller to `.await` it, an unbuildable
    /// result strictly worse than the pre-existing, order-dependent behavior this pass
    /// replaces). A real multi-hop socket chain (this seed's actual target) never runs through
    /// a struct method this way, since nothing here currently lets a method hold an open
    /// connection socket across the call.
    const TOKIO_ASYNC_NETWORK_TYPES: &[&str] = &[
        "TcpListener", "TcpStream", "UdpSocket", "UnixListener", "UnixStream", "UnixDatagram",
    ];

    fn fn_body_has_direct_tokio_async_call(f: &FnDecl) -> bool {
        let dual_params: std::collections::HashSet<&str> = f.params.iter()
            .filter_map(|p| match p.ty.as_ref()?.without_mut() {
                Type::Named(n) if Self::TOKIO_ASYNC_NETWORK_TYPES.contains(&n.as_str()) => Some(p.name.as_str()),
                _ => None,
            })
            .collect();
        let mut found = false;
        crate::ast::walk_stmts(&f.body, &mut |node| {
            if found { return; }
            let crate::ast::WalkNode::Expr(Expr { kind: ExprKind::MethodCall(recv, method, _), .. }) = node else { return };
            let ExprKind::Var(recv_name) = &recv.kind else { return };
            let rust_method = camel_to_snake(method);

            // Instance call on a parameter declared with a dual sync/async type.
            if dual_params.contains(recv_name.as_str())
                && Self::TOKIO_ASYNC_METHODS.contains(&rust_method.as_str())
            {
                found = true;
            }
        });
        found
    }
}
