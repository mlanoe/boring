// Unit tests of the transpiler, split out of `mod.rs` (see `mod tests;` there).
// Indentation is kept from the old inline module: several cases embed multi-line
// Boring sources whose indentation is significant.
    use super::*;

    fn transpile_src_with_config(src: &str, config: TranspileConfig) -> String {
        let tokens = crate::lexer::lex(src).expect("lex error");
        let program = crate::parser::parse(tokens).expect("parse error");
        transpile_with_config(&program, config).code
    }

    /// Transpiler error messages (not the generated code) for `src`, joined by newlines.
    fn transpile_src_errors(src: &str) -> String {
        let tokens = crate::lexer::lex(src).expect("lex error");
        let program = crate::parser::parse(tokens).expect("parse error");
        transpile_with_config(&program, TranspileConfig::default()).errors
            .iter().map(|e| e.message.clone()).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn string_methods_on_indexed_string_values_use_instance_dispatch() {
        let src = r#"
def main() throws:
    let text = " Hello World\nsecond "
    let [string] values = [" Mixed Case "]
    let a = text.split("\n")[0].split(" ")
    let b = text.split("\n")[0].trim()
    let c = text.split("\n")[0].replace("World", "there")
    let d = text.split("\n")[0].startsWith(" Hello")
    let e = text.split("\n")[0].toLower()
    let f = values[0].trim()
    print "{a[1]} {b} {c} {d} {e} {f}"
"#;
        for threading in [ThreadingMode::Multi, ThreadingMode::Single] {
            let code = transpile_src_with_config(src, TranspileConfig {
                threading,
                ..TranspileConfig::default()
            });
            assert!(!code.contains("]::split("), "got:\n{code}");
            assert!(!code.contains("]::trim("), "got:\n{code}");
            assert!(!code.contains("]::replace("), "got:\n{code}");
            assert!(!code.contains("]::startsWith("), "got:\n{code}");
            assert!(!code.contains("]::toLower("), "got:\n{code}");
            assert!(code.contains(".clone().split(\" \")"), "got:\n{code}");
            assert!(code.contains(".clone().starts_with(\" Hello\")"), "got:\n{code}");
            assert!(code.contains(".clone().to_lowercase()"), "got:\n{code}");
        }
    }

    #[test]
    fn string_methods_on_inferred_case_conversion_locals_stay_owned() {
        let src = r#"
def string direct(string s):
    return s.replace("x", " ").trim()

def main() throws:
    let outside = " X ".toLower()
    let outside_value = outside.replace("x", "").trim()
    for line in " A ".split(" "):
        let low = line.toLower()
        let value = low.replace("a", " A ").trim()
        print value
    let lines = [" b "]
    var i = 0
    while i < lines.length:
        let upper = lines[i].toUpper()
        let value = upper.replace("B", " B ").trimStart().trimEnd()
        print value
        i += 1
    print outside_value.trim().trim()
    print direct(" x ")
"#;
        for (threading, str_ty) in [
            (ThreadingMode::Multi, "Arc::<str>"),
            (ThreadingMode::Single, "Rc::<str>"),
        ] {
            let code = transpile_src_with_config(src, TranspileConfig {
                threading,
                ..TranspileConfig::default()
            });
            assert!(code.contains(&format!(
                "let value = {str_ty}::from({str_ty}::from(low.replace(\"a\", \" A \").as_str()).trim());"
            )), "got:\n{code}");
            assert!(code.contains(&format!(
                "let value = {str_ty}::from({str_ty}::from({str_ty}::from(upper.replace(\"B\", \" B \").as_str()).trim_start()).trim_end());"
            )), "got:\n{code}");
            assert!(code.contains(&format!(
                "{str_ty}::from({str_ty}::from(outside_value.trim()).trim())"
            )), "got:\n{code}");
            assert!(!code.contains("let value = low.replace(\"a\", \" A \").trim();"), "got:\n{code}");
        }
    }

    #[test]
    fn explicit_borrow_receivers_keep_nested_field_types_for_numeric_casts() {
        let src = r#"
struct Inner:
    float32 t

struct Outer:
    Inner d
    [Inner] items

    req Inner inner():
        d

def float by_borrow(Outer& o):
    o.d.t as float

def float by_mut_borrow(mut Outer& o):
    o.d.t as float

def float by_var_borrow(var Outer& o):
    o.d.t as float

def float by_alias(Outer& o):
    let Outer& alias = o
    alias.d.t as float

def float by_method(Outer& o):
    o.inner().t as float

def int by_index(Outer& o, int i):
    o.items[i].t as int
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("(o.d.t as f64)"), "got:\n{code}");
        assert!(code.contains("(alias.d.t as f64)"), "got:\n{code}");
        assert!(code.contains("(o.inner().t as f64)"), "got:\n{code}");
        assert!(code.contains(".t as isize)"), "got:\n{code}");
        assert!(!code.contains(".trim().parse::<f64>()"), "got:\n{code}");
        assert!(!code.contains(".trim().parse::<isize>()"), "got:\n{code}");
    }

    #[test]
    fn host_tensor_calls_lower_to_portable_rust_loops() {
        let src = "let [float32, k = 5, m = 3]'kernel'global a = [float32(i) for i in 0..<15]\nlet [float32, n = 7, k = 5]'kernel'unified b = [float32(i) for i in 0..<35]\nmut [float32, n = 7, m = 3]'kernel'unified c = [float32(0) for ..<21]\nkernel.tensors.matmul(a, b, c)\nkernel.tensors.mma(a, b, c)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert_eq!(code.matches("for __boring_tensor_row in 0usize..3usize").count(), 2);
        assert!(code.contains("for __boring_tensor_col in 0usize..7usize"));
        assert!(code.contains("for __boring_tensor_k in 0usize..5usize"));
        assert!(code.contains("let mut __boring_tensor_sum: f32 = 0.0f32"));
        assert!(code.contains("let mut __boring_tensor_sum: f32 = c["));
        assert!(!code.contains("gpu::tensor"));
    }

    #[test]
    fn host_tensor_linear_uses_row_major_weights() {
        let src = "let [float32, k = 5, m = 3]'kernel'global x = [float32(i) for i in 0..<15]\nlet [float32, k = 5, n = 7]'kernel'global w = [float32(i) for i in 0..<35]\nmut [float32, n = 7, m = 3]'kernel'unified y = [float32(0) for ..<21]\nkernel.tensors.linear(x, w, y)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("w[__boring_tensor_col * 5 + __boring_tensor_k]"), "{code}");
    }

    #[test]
    fn test_managed_multi_wraps_owned() {
        // T'owned → Arc<Mutex<T>> in managed+multi; plain Named is NOT wrapped.
        let src = "struct Counter:\n    init(pub int n)\ndef Counter'owned make(): Counter(n = 5)\n";
        let config = TranspileConfig { mode: TranspileMode::Managed, threading: ThreadingMode::Multi, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("Arc<std::sync::Mutex<Counter>>"),
            "T'owned should become Arc<std::sync::Mutex> in managed+multi:\n{}", code);
    }

    #[test]
    fn test_managed_single_wraps_owned() {
        // T'owned → RefCell<T> in managed+single; plain Named is NOT wrapped.
        let src = "struct Counter:\n    init(pub int n)\ndef Counter'owned make(): Counter(n = 5)\n";
        let config = TranspileConfig { mode: TranspileMode::Managed, threading: ThreadingMode::Single, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("RefCell<Counter>"),
            "T'owned should become RefCell in managed+single:\n{}", code);
    }

    #[test]
    fn test_task_method_mutating_body_gets_mut_self() {
        // A `task`-qualified struct method declared inline in the struct body
        // (`f.qualifier = None`) that mutates a field must emit `&mut self`, exactly
        // like a plain `def` method — it is only an *external* `task T.method()`
        // declaration (called via Arc<Self>, `f.qualifier = Some(_)`) that is
        // restricted to `&self`. Regression test for a bug where all `task` methods
        // were forced to `&self` regardless of `qualifier`, producing invalid Rust
        // (E0594 "cannot assign to `self.count`, which is behind a `&` reference").
        let src = "\
struct Widget:\n    var int count\n\n    task void doAsyncThing():\n        count = count + 1\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("async fn doAsyncThing(&mut self)"),
            "a mutating inline task method must emit &mut self, got:\n{}", code);
    }

    #[test]
    fn test_task_req_method_stays_ref_self() {
        // Companion to the above: a non-mutating `task req` method must still emit
        // the plain `&self` it already got before the fix (no regression on the
        // read-only case).
        let src = "\
struct Widget:\n    var int count\n\n    task req int fetchCount():\n        count\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("async fn fetchCount(&self)"),
            "a non-mutating task req method must keep &self, got:\n{}", code);
    }

    // ── emit_task_await / emit_plain_task_await ─────────────────────────────
    //
    // Regression tests for factoring the `.value`/`.wait` await dispatch (used to
    // be duplicated inline at 3+ call sites) into two shared helpers. Each test
    // below pins one call site's exact output so the refactor can't silently
    // collapse two historically-distinct branches into one.

    #[test]
    fn test_throws_join_handle_wait_in_throws_ctx_maps_join_error() {
        // task_vars path, is_throws_handle=true, use_error_mapping=true: a
        // throws-context `.wait` on a JoinHandle<Result<T, BoringError>> must
        // map_err the JoinError into a boxed error before the double `?`.
        let src = "\
task def int work() throws: 42\n\ndef run() throws:\n    let f = task work()\n    f.wait\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("map_err(|__e| Box::new(BoringError::String"),
            "throws-JoinHandle wait in a throws context must map_err the JoinError, got:\n{}", code);
        assert!(code.contains(")??; }"),
            "throws-JoinHandle wait in a throws context must double-`?` past both errors, got:\n{}", code);
    }

    #[test]
    fn test_timeout_join_handle_wait_in_throws_ctx_skips_error_mapping() {
        // TaskWithTimeout path, is_throws_handle=true, use_error_mapping=false:
        // unlike the throws-JoinHandle case above, its inner error (`Elapsed`)
        // already converts into the throws context's own error type, so no
        // map_err/BoringError dance is needed here — just expect + `?`. This is
        // the one branch where the two "throws handle" call sites genuinely
        // differ, so collapsing them into one path with no distinction would be
        // a real regression.
        // Uses the INLINE form (`.wait` chained directly off the `task(...)`
        // expression, never bound to a `let`) — a `let`-bound TaskWithTimeout
        // is tracked in task_vars and goes through the *other* helper call site
        // (also is_throws_handle=true, but with use_error_mapping=true, since
        // task_vars can't distinguish a real `throws` fn's JoinHandle from a
        // timeout's); only the truly inline form reaches this one.
        let src = "\
def int fetch(int n):\n    n * 10\n\ndef run() throws:\n    (task(Duration.from_millis(50)): fetch(5)).wait\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(!code.contains("map_err(|__e| Box::new(BoringError::String"),
            "timeout-JoinHandle wait must NOT use the throws-JoinHandle's error mapping, got:\n{}", code);
        assert!(code.contains(".await.expect(\"task panicked\")?"),
            "timeout-JoinHandle wait in a throws context should be a plain expect + `?`, got:\n{}", code);
    }

    #[test]
    fn test_tail_wait_on_inline_task_timeout_promotes_fn_to_async() {
        // Regression test: a `throws` fn whose only async content is a bare
        // `(task(dur): body).wait`/`.value` in TAIL position never got promoted to
        // `async fn` — `body_has_channel_or_task`/`expr_has_channel_or_task` (which
        // `implicit_async` in emit_top.rs consults) only matched `ExprKind::Task`/
        // `TaskWithTimeout` at the top level of a statement, not one level down
        // behind the `.wait`/`.value` field access that actually wraps it here.
        // The emitted `.await` then landed inside a non-`async fn`, an E0728
        // compile error ("await is only allowed inside async functions and
        // blocks") never caught by the interpreter-only test suite. Fixed by
        // having both helpers recurse through `ExprKind::Field`/`OptionalField`.
        let src = "\
def int fetch(int n):\n    n * 10\n\ndef run() throws:\n    (task(Duration.from_millis(50)): fetch(5)).wait\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("async fn run("),
            "a throws fn with a tail `.wait` on an inline task(dur) must be promoted to `async fn`, got:\n{}", code);
    }

    #[test]
    fn test_async_propagates_transitively_through_plain_callers() {
        // Regression test: `implicit_async` in emit_top.rs used to only check
        // "does THIS function's own body call an already-async function" for
        // `def main():` (the `main_needs_async` special case) — a plain function
        // calling another plain function that is itself implicitly async (because
        // ITS body spawns a `task:`) stayed a plain `fn`, so the `.await` the
        // call site needed (gated on `self.in_async`) was never emitted, and a
        // `throws` call got a bare `?` applied directly to a `Future` instead
        // (E0277 "the `?` operator can only be applied to values that implement
        // `Try`"). Found via a real boring-llm server combining `task:`-based
        // connection handling with a plain `run_serve` wrapper calling `serve`,
        // called in turn from `main` — three hops, none of them `main` itself at
        // the innermost two. Fixed by checking `body_calls_task_fn` against the
        // accumulating `task_fns` set for every free function, not just `main`.
        let src = "\
def helper() throws:\n    task: print \"hi\"\n\ndef wrapper() throws:\n    helper()\n    print \"wrapper done\"\n\ndef main() throws:\n    wrapper()\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("async fn helper("), "helper() itself spawns a task, got:\n{}", code);
        assert!(code.contains("async fn wrapper("),
            "wrapper() calls the now-async helper() and must become async too, got:\n{}", code);
        assert!(code.contains("helper().await?"),
            "wrapper()'s call to helper() must be awaited (and `?`-propagated, since helper() throws), got:\n{}", code);
        assert!(code.contains("#[tokio::main]") && code.contains("async fn main("),
            "main() transitively calls a task-spawning function (through wrapper()) and must get the tokio entry point, got:\n{}", code);
        assert!(code.contains("wrapper().await?"),
            "main()'s call to wrapper() must be awaited too, got:\n{}", code);
    }

    #[test]
    fn test_plain_inline_task_value_and_wait_ignore_throws_context() {
        // emit_plain_task_await path (inline `(task: expr).value`/`.wait`, whose
        // `is_future` heuristic matches `ExprKind::Task(_)` directly): always
        // `.expect`s / discards, regardless of `throws` context — there's no
        // tracked inner-Result shape to propagate here, unlike the two
        // task_vars-backed paths above. Uses a captureless task body (`task: 42`)
        // to avoid the separate non-Arc-capture inlining special case (see
        // `emit_expr_field`'s "Special case" comment), which would otherwise
        // intercept `.value`/`.wait` before this heuristic is even reached.
        let src = "\
def run() throws:\n    let v = (task: 42).value\n    print v\n    (task: 42).wait\n    print \"done\"\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains(".await.expect(\"task panicked\")") && !code.contains(".await.expect(\"task panicked\")?"),
            "inline task .value must always .expect, never `?` even in a throws context, got:\n{}", code);
        assert!(code.contains("{ let _ = ") && code.contains(".await; }"),
            "inline task .wait must discard the awaited value unconditionally, got:\n{}", code);
    }

    #[test]
    fn test_managed_named_not_wrapped() {
        // Plain Named (Type::Named) is NOT wrapped in managed mode — only T'owned is.
        let src = "struct Counter:\n    init(pub int n)\n\nlet Counter c = Counter(n = 0)\n";
        let config = TranspileConfig { mode: TranspileMode::Managed, threading: ThreadingMode::Multi, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(!code.contains("Arc<tokio::sync::Mutex<Counter>>"),
            "managed mode: plain Named should NOT be wrapped, got:\n{}", code);
    }

    #[test]
    fn test_strict_mode_no_managed_wrapping() {
        let src = "struct Counter:\n    init(pub int n)\n\nlet Counter c = Counter(n = 0)\n";
        let config = TranspileConfig { mode: TranspileMode::Strict, threading: ThreadingMode::Multi, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(!code.contains("Arc<tokio::sync::Mutex<Counter>>"),
            "strict mode: Counter should NOT be wrapped, got:\n{}", code);
    }

    #[test]
    fn test_managed_unit_enum_not_wrapped() {
        // Non-parametric enums must remain Copy — never wrapped in managed mode.
        let src = "enum Color:\n    Red\n    Green\n    Blue\n\nlet Color c = Color.Red\n";
        let config = TranspileConfig { mode: TranspileMode::Managed, threading: ThreadingMode::Multi, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(!code.contains("Arc<tokio::sync::Mutex<Color>>"),
            "managed mode: unit enum should NOT be wrapped, got:\n{}", code);
    }

    #[test]
    fn test_let_generic_struct_ctor_no_bare_type_annotation() {
        // `let b = Container(value = 42, tag = 1)` with no explicit Boring-side type
        // annotation must NOT emit a bare `let b: Container = ...` — Container<T> needs
        // a generic argument Rust can't recover from the struct name alone (E0107,
        // "missing generics for struct `Container`"). Either omit the annotation
        // entirely (let Rust infer `Container<i64>` from the RHS struct literal) or
        // emit the fully-applied `Container<i64>` — never the bare, arity-mismatched name.
        let src = "\
struct Container<T>:\n    T value\n    int tag\n\n\
def main():\n    let b = Container(value = 42, tag = 1)\n    print \"{b.value}\"\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(!code.contains(": Container ="),
            "generic struct ctor must not get a bare, generics-less `let` type annotation:\n{}", code);
        assert!(code.contains("let b = Container"),
            "expected the annotation to be omitted so Rust infers Container<i64> from the RHS:\n{}", code);
    }

    #[test]
    fn test_for_loop_destructure_marks_mut_tuple_slot() {
        // `Query<(mut Position&, Velocity&)>`-shaped parameter (Bevy-ECS-style,
        // see docs/book.md's "Advanced — Explicit borrow syntax: `T&`"): destructuring
        // its items in a `for` loop must bind the slot declared `mut T&` with
        // a Rust `mut` in the pattern — required for that item's type (e.g.
        // Bevy's `Mut<T>` change-detection wrapper) to be field-mutable, since
        // Boring has no source syntax to write `mut` on a for-loop variable
        // directly (only inferred from the iterable's declared type).
        let src = "\
struct Position:\n    pub var float x\n\nstruct Velocity:\n    pub float x\n\n\
def move_system(Query<(mut Position&, Velocity&)> query):\n    for pos, vel in query:\n        pos.x = pos.x + vel.x\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("for (mut pos, vel) in"),
            "expected `mut` on the slot declared `mut Position&`, got:\n{}", code);
    }

    #[test]
    fn test_for_loop_destructure_no_spurious_mut() {
        // Non-regression: a plain (non-`mut`-tagged) tuple/dict destructure
        // must NOT gain a `mut` binding it didn't have before this feature.
        let src = "let m = {\"a\" = 1}\nfor k, v in m:\n    print \"{k}={v}\"\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        // Check the actual `for` line, not a substring match anywhere in the
        // file — the standard-library boilerplate Boring always emits (e.g.
        // `BoringArrayIndex::remove_at`) legitimately contains `let mut v = ...`
        // elsewhere, which a bare `!code.contains("mut v")` would false-positive on.
        assert!(code.contains("for (k, v) in"),
            "plain dict destructure must not gain a spurious `mut`, got:\n{}", code);
    }

    #[test]
    fn test_for_loop_destructure_mixed_mut_slots() {
        // Both tuple positions independently `mut`-tagged, or neither, must
        // each be judged on their own slot — not "any mut anywhere -> all mut".
        let src = "\
struct A:\n    pub var float x\n\nstruct B:\n    pub var float x\n\n\
def sys(Query<(mut A&, mut B&)> query):\n    for a, b in query:\n        a.x = a.x + b.x\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("for (mut a, mut b) in"),
            "both slots declared `mut T&` should both bind `mut`, got:\n{}", code);
    }

    #[test]
    fn test_component_and_resource_auto_derive() {
        let src = "component Position:\n    pub float x\n    pub float y\n\ncomponent Frozen: pass\n\nresource Score:\n    pub var int points = 0\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("Component"), "expected Component derive on Position, got:\n{}", code);
        assert!(code.contains("struct Position"), "expected struct Position, got:\n{}", code);
        assert!(code.contains("struct Frozen"), "expected struct Frozen, got:\n{}", code);
        assert!(code.contains("Resource"), "expected Resource derive on Score, got:\n{}", code);
        assert!(code.contains("struct Score"), "expected struct Score, got:\n{}", code);
        assert!(code.contains("use bevy::prelude::*;"), "expected bevy prelude import, got:\n{}", code);
    }

    #[test]
    fn test_query_transpile_movement() {
        let src = "\
component Position:
    pub float x
    pub float y

component Velocity:
    pub float dx
    pub float dy

component Frozen: pass

def movement(mut Position pos, Velocity vel) where not Frozen:
    pos.x += vel.dx * .time.delta
    pos.y += vel.dy * .time.delta
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("pub fn movement("), "expected pub fn movement, got:\n{}", code);
        assert!(code.contains("Query<(&mut Position, &Velocity), Without<Frozen>>"), "expected Query type, got:\n{}", code);
        assert!(code.contains("time: Res<Time>"), "expected time: Res<Time>, got:\n{}", code);
        assert!(code.contains("for (mut pos, vel) in &mut query"), "expected for loop, got:\n{}", code);
        assert!(code.contains("pos.x += (vel.dx * (time.delta_secs() as f64))"), "expected pos.x update, got:\n{}", code);
    }

    #[test]
    fn test_query_transpile_startup_spawn() {
        let src = "\
component Position:
    pub float x
    pub float y

@schedule(Startup)
def setup_world():
    .spawn(Position(x = 1.0, y = 2.0))
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("// @schedule(Startup)"), "expected @schedule(Startup) comment, got:\n{}", code);
        assert!(code.contains("pub fn setup_world(mut commands: Commands)"), "expected pub fn setup_world with commands, got:\n{}", code);
        assert!(code.contains("commands.spawn("), "expected commands.spawn, got:\n{}", code);
    }

    #[test]
    fn test_query_transpile_despawn() {
        let src = "\
component Health:
    pub int current

@schedule(Update)
def death_check(Health h):
    .despawn()
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("pub fn death_check("), "expected pub fn death_check, got:\n{}", code);
        assert!(code.contains("Query<(Entity, &Health)>"), "expected Query with Entity, got:\n{}", code);
        assert!(code.contains("mut commands: Commands"), "expected mut commands: Commands, got:\n{}", code);
        assert!(code.contains("for (__entity, h) in &mut query"), "expected for loop with __entity, got:\n{}", code);
        assert!(code.contains("commands.entity(__entity).despawn()"), "expected despawn call, got:\n{}", code);
    }

    #[test]
    fn test_data_component_method_query() {
        let src = "\
@component
data Position:
    float x
    float y

    @schedule(Update)
    def movement(Velocity vel) where not Frozen:
        self.x += vel.dx * .time.delta
        self.y += vel.dy * .time.delta

@component
data Velocity:
    float dx
    float dy

@component
data Frozen: pass
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("pub fn position_movement("), "expected position_movement fn, got:\n{}", code);
        assert!(code.contains("Query<(&mut Position, &Velocity), Without<Frozen>>"), "expected Query type, got:\n{}", code);
        assert!(code.contains("time: Res<Time>"), "expected time: Res<Time>, got:\n{}", code);
        assert!(code.contains("for (mut self_, vel) in &mut query"), "expected for loop, got:\n{}", code);
        assert!(code.contains("self_.x += (vel.dx * (time.delta_secs() as f64));"), "expected self_.x update, got:\n{}", code);
    }

    #[test]
    fn test_resource_and_query_mutation() {
        let src = "\
@resource
var int score = 0

@component
data Position:
    float x
    float y

@schedule(Update)
def add_score(Position pos):
    score += 10
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("#[derive(bevy::prelude::Resource, Debug, Clone, PartialEq)]"), "expected derive Resource, got:\n{}", code);
        assert!(code.contains("pub struct Res_score(pub isize);"), "expected Res_score struct, got:\n{}", code);
        assert!(code.contains("mut res_score: ResMut<Res_score>"), "expected ResMut<Res_score>, got:\n{}", code);
        assert!(code.contains("res_score.0 = res_score.0.wrapping_add((10 as isize));"), "expected res_score update, got:\n{}", code);
    }

    #[test]
    fn test_query_dot_resource_access() {
        let src = "\
@resource
data LayerOrders:
    dict<string, int> ranks

    req int get(string tid):
        self.ranks[tid]

@resource
var int score = 0

@component
data Transform:
    float z

@component
data ScratchSprite:
    string target_id

@schedule(Update)
def sync_layers(mut Transform transform, ScratchSprite tag):
    let order = .resource.layerOrders.get(tag.target_id)
    transform.z = order as float

@schedule(Update)
def increment_score():
    .resource.score += 1
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        // sync_layers should take Query<(&mut Transform, &ScratchSprite)> and res_layer_orders: Res<LayerOrders>
        assert!(code.contains("pub fn sync_layers(mut query: Query<(&mut Transform, &ScratchSprite)>, res_layer_orders: Res<LayerOrders>)"),
            "expected sync_layers signature, got:\n{}", code);
        assert!(code.contains("let order = res_layer_orders.get("),
            "expected res_layer_orders.get, got:\n{}", code);

        // increment_score should take mut res_score: ResMut<Res_score> and NO query loop
        assert!(code.contains("pub fn increment_score(mut res_score: ResMut<Res_score>)"),
            "expected increment_score signature, got:\n{}", code);
        assert!(code.contains("res_score.0 += 1;"),
            "expected res_score += 1, got:\n{}", code);
    }

    #[test]
    fn test_ecs_resource_in_def_generates_context_and_reborrows() {
        let src = "\
@resource
data Vars:
    [int] globals

    def void set(int slot, int val):
        self.globals[slot] = val

def step_thread(int slot):
    ecs.resource.vars.set(slot, 42)

@schedule(Update)
def run_threads():
    step_thread(0)
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("pub struct EcsContext<'a> {"), "expected EcsContext struct in:\n{}", code);
        assert!(code.contains("pub vars: &'a mut Vars,"), "expected pub vars in EcsContext:\n{}", code);
        assert!(code.contains("fn step_thread(slot: isize, ecs: &mut EcsContext)"), "expected ecs param in step_thread:\n{}", code);
        assert!(code.contains("ecs.vars.set("), "expected ecs.vars.set call:\n{}", code);
        assert!(code.contains("step_thread(0, &mut EcsContext { vars: &mut *res_vars });"), "expected passing &mut EcsContext in query call:\n{}", code);
    }

    #[test]
    fn test_ecs_resource_subscript_in_def() {
        let src = "\
@resource
data Vars:
    [int] globals

    req int get(int slot):
        return self.globals[slot]

    def set(int slot, int val):
        self.globals[slot] = val

def step_thread(int slot):
    ecs.resource.vars[slot] = 42
    let x = ecs.resource.vars[slot]

@schedule(Update)
def run_threads():
    step_thread(0)
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("ecs.vars.set("), "expected ecs.vars.set, got:\n{}", code);
        assert!(code.contains("let x = ecs.vars.get("), "expected ecs.vars.get, got:\n{}", code);
    }

    #[test]
    fn test_ecs_context_local_provider_stops_propagation() {
        let src = "\
@resource
data Vars:
    [int] globals

    def void set(int slot, int val):
        self.globals[slot] = val

def step_thread(int slot):
    ecs.resource.vars.set(slot, 42)

def run_headless():
    var vars = Vars(globals = [0, 0, 0])
    var ecs = EcsContext(vars = vars)
    step_thread(0)

def test_entry():
    run_headless()
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn step_thread(slot: isize, ecs: &mut EcsContext)"), "step_thread needs ecs:\n{}", code);
        assert!(code.contains("fn run_headless()"), "run_headless should NOT need ecs param:\n{}", code);
        assert!(!code.contains("fn run_headless(ecs:"), "run_headless should NOT have ecs param:\n{}", code);
        assert!(code.contains("fn test_entry()"), "test_entry should NOT need ecs param:\n{}", code);
        assert!(!code.contains("fn test_entry(ecs:"), "test_entry should NOT have ecs param:\n{}", code);
        assert!(code.contains("let mut ecs = EcsContext { vars: &mut vars };"), "expected EcsContext construction:\n{}", code);
        assert!(code.contains("step_thread(0, &mut ecs);"), "expected step_thread call passing ecs:\n{}", code);
    }

    #[test]
    fn test_query_many_resources_grouped_in_tuples() {
        let mut src = String::new();
        for i in 0..18 {
            src.push_str(&format!("@resource\ndata Res{i}:\n    int v\n\n"));
        }
        src.push_str("@schedule(Update)\ndef big_system():\n");
        for i in 0..18 {
            src.push_str(&format!("    let _ = .resource.res{i}\n"));
        }
        let code = transpile_src_with_config(&src, TranspileConfig::default());
        // Since there are 18 resources, Bevy's 16-parameter limit is exceeded, so boring chunks them into tuples
        assert!(code.contains("pub fn big_system(("), "expected tuple parameter grouping in:\n{}", code);
    }

    #[test]
    fn test_dot_resource_outside_query_is_compiler_error() {
        let src = "\
@resource
data Vars:
    [int] globals

def step_thread():
    let x = .resource.vars
";
        let errs = transpile_src_errors(src);
        assert!(errs.contains("`.resource` is only allowed inside ECS systems"),
            "expected error for .resource outside ECS system, got:\n{}", errs);
    }

    #[test]
    fn test_query_resource_passed_to_mut_fn() {
        let src = "\
@resource
data Vars:
    int value

@resource
data Pending:
    [int] queue

def mutate_vars(mut Vars vars):
    vars.value = 100

def read_vars(Vars vars):
    let v = vars.value

@schedule(Update)
def run_system():
    mutate_vars(.resource.vars)

@schedule(Update)
def run_system_read():
    read_vars(.resource.vars)

@schedule(Update)
def run_system_method():
    .resource.pending.queue.reverse()
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("pub fn run_system(mut res_vars: ResMut<Vars>)"),
            "expected ResMut<Vars>, got:\n{}", code);
        assert!(code.contains("pub fn run_system_read(res_vars: Res<Vars>)"),
            "expected Res<Vars>, got:\n{}", code);
        assert!(code.contains("pub fn run_system_method(mut res_pending: ResMut<Pending>)"),
            "expected ResMut<Pending>, got:\n{}", code);
    }

    #[test]
    fn test_query_system_param_separation() {
        let src = "\
@resource
data Score:
    int value

@component
data Position:
    float x

def handle_events(mut Commands commands, Res<Score> score, mut Position pos):
    pos.x += score.value as float
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        // Commands and Res<Score> should NOT be inside Query<...>!
        assert!(code.contains("mut query: Query<&mut Position>"),
            "expected Query<&mut Position>, got:\n{}", code);
        assert!(code.contains("mut commands: Commands"),
            "expected mut commands: Commands, got:\n{}", code);
        assert!(code.contains("score: Res<Score>"),
            "expected score: Res<Score>, got:\n{}", code);
        assert!(code.contains("for mut pos in &mut query"),
            "expected for mut pos in &mut query, got:\n{}", code);
    }

    #[test]
    fn test_game_bootstrap_default_camera() {
        let src = "\
@game
data Breakout:
    pass

@schedule(Startup)
def setup(mut Commands commands):
    pass

@schedule(Update)
def move_paddle(mut Commands commands):
    pass
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("pub type Camera2D = Camera2d;"), "expected Camera2D alias, got:\n{}", code);
        assert!(code.contains("fn __boring_default_camera(mut commands: Commands)"), "expected default camera system, got:\n{}", code);
        assert!(code.contains("app.add_plugins(DefaultPlugins.set(WindowPlugin {"), "expected WindowPlugin, got:\n{}", code);
        assert!(code.contains("title: \"Breakout\".into()"), "expected title Breakout, got:\n{}", code);
        assert!(code.contains("resolution: (800u32, 600u32).into()"), "expected default resolution, got:\n{}", code);
        assert!(code.contains("app.add_systems(Startup, (__boring_default_camera, setup));"), "expected Startup systems with default camera, got:\n{}", code);
        assert!(code.contains("app.add_systems(Update, move_paddle);"), "expected Update systems, got:\n{}", code);
        assert!(code.contains("build_app().run();"), "expected build_app().run(), got:\n{}", code);
    }

    #[test]
    fn test_game_custom_window_and_explicit_camera() {
        let src = "\
@game
data Pong:
    string title = \"Super Pong\"
    int width = 1024
    int height = 768
    Color clear_color = Color.srgb(0.1, 0.1, 0.1)

@resource
data Score:
    int count = 0

@schedule(Startup)
def spawn_camera(mut Commands commands):
    commands.spawn(Camera2D())

@schedule(Update)
@after(spawn_camera)
def update_score(mut ResMut<Score> score):
    score.count += 1
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("title: \"Super Pong\".into()"), "expected title Super Pong, got:\n{}", code);
        assert!(code.contains("resolution: (1024u32, 768u32).into()"), "expected custom resolution, got:\n{}", code);
        assert!(code.contains("app.insert_resource(ClearColor("), "expected ClearColor, got:\n{}", code);
        assert!(code.contains("app.init_resource::<Score>();"), "expected init_resource Score, got:\n{}", code);
        assert!(code.contains("app.init_resource::<Pong>();"), "expected init_resource Pong, got:\n{}", code);
        // Explicit camera was spawned -> __boring_default_camera should NOT be generated!
        assert!(!code.contains("__boring_default_camera"), "expected no __boring_default_camera, got:\n{}", code);
        assert!(code.contains("app.add_systems(Startup, spawn_camera);"), "expected Startup spawn_camera, got:\n{}", code);
        assert!(code.contains("update_score.after(spawn_camera)"), "expected update_score.after(spawn_camera), got:\n{}", code);
    }

    #[test]
    fn test_transform_convenience_access() {
        let src = "\
@component
data Velocity:
    float x = 0.0

@schedule(Update)
def move_paddle(mut Transform transform, Velocity velocity):
    transform.x = transform.x + 10.0
    let cur_y = transform.y
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("transform.translation.x"),
            "expected transform.translation.x in code, got:\n{}", code);
        assert!(code.contains("transform.translation.y"),
            "expected transform.translation.y read, got:\n{}", code);
    }

    #[test]
    fn test_resource_params_auto_wrapped() {
        let src = "\
@resource
data Score:
    int count = 0

@schedule(Startup)
def setup_game(mut Commands commands, AssetServer asset_server, mut Assets<Mesh> meshes):
    pass

@schedule(Update)
def update_score(Score score, mut Score mut_score):
    pass
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn setup_game(mut commands: Commands, asset_server: Res<AssetServer>, mut meshes: ResMut<Assets<Mesh>>)"),
            "expected setup_game with wrapped resources, got:\n{}", code);
        assert!(code.contains("fn update_score(score: Res<Score>, mut mut_score: ResMut<Score>)"),
            "expected update_score with wrapped resources, got:\n{}", code);
    }

    #[test]
    fn test_ecs_use_mapping_and_enriched_prelude() {
        let src = "\
use ecs.input.KeyboardInput
use ecs.render.Extent3d
use ecs.audio.Volume

@schedule(Startup)
def init_game(mut Commands commands):
    pass
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("use bevy::prelude::*;"), "expected bevy::prelude::*, got:\n{}", code);
        assert!(code.contains("use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};"),
            "expected Extent3d in prelude, got:\n{}", code);
        assert!(code.contains("use bevy::input::keyboard::KeyboardInput;"),
            "expected KeyboardInput in prelude, got:\n{}", code);
        assert!(code.contains("use bevy::audio::Volume;"),
            "expected Volume in prelude, got:\n{}", code);
    }

    #[test]
    fn test_vec2_vec3_array_coercion() {
        let src = "\
pub let Vec2 PADDLE_SIZE = [120.0, 20.0]
pub let Vec3 BALL_START = [0.0, -50.0, 1.0]

@schedule(Startup)
def setup(mut Commands commands):
    let Vec2 v2 = [1.0, 2.0]
    let Vec3 v3 = [1.0, 2.0, 3.0]
    let Vec2? opt_v2 = [4.0, 5.0]
    let t = Transform(translation = [1.0, 2.0, 3.0], scale = [4.0, 5.0, 6.0], _)
    let s = Sprite(custom_size = [10.0, 20.0], _)
    let a = Anchor([0.0, -1.0])
";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("pub const PADDLE_SIZE: Vec2 = Vec2::new((120.0) as f32, (20.0) as f32);"),
            "expected const PADDLE_SIZE, got:\n{}", code);
        assert!(code.contains("pub const BALL_START: Vec3 = Vec3::new((0.0) as f32, (-50.0) as f32, (1.0) as f32);")
            || code.contains("pub const BALL_START: Vec3 = Vec3::new((0.0) as f32, ((-50.0)) as f32, (1.0) as f32);"),
            "expected const BALL_START, got:\n{}", code);
        assert!(code.contains("let v2: Vec2 = Vec2::new((1.0) as f32, (2.0) as f32);"),
            "expected v2, got:\n{}", code);
        assert!(code.contains("let v3: Vec3 = Vec3::new((1.0) as f32, (2.0) as f32, (3.0) as f32);"),
            "expected v3, got:\n{}", code);
        assert!(code.contains("let opt_v2: Option<Vec2> = Some(Vec2::new((4.0) as f32, (5.0) as f32));"),
            "expected opt_v2, got:\n{}", code);
        assert!(code.contains("translation: Vec3::new((1.0) as f32, (2.0) as f32, (3.0) as f32)"),
            "expected translation, got:\n{}", code);
        assert!(code.contains("scale: Vec3::new((4.0) as f32, (5.0) as f32, (6.0) as f32)"),
            "expected scale, got:\n{}", code);
        assert!(code.contains("custom_size: Some(Vec2::new((10.0) as f32, (20.0) as f32))"),
            "expected custom_size, got:\n{}", code);
        assert!(code.contains("Anchor(Vec2::new((0.0) as f32, ((-1.0)) as f32))"),
            "expected Anchor, got:\n{}", code);
    }

    #[test]
    fn test_for_loop_over_unannotated_enum_array_preserves_camel_case_method() {
        // Regression: a `for` loop's single iteration variable over an *unannotated*
        // `let arr = [EnumType.Variant(...), ...]` array literal never got its element
        // type recorded in `var_types` (only an explicit `let [EnumType] arr = ...`
        // annotation did, via the `s.ty` branch in emit_let.rs). Without that type,
        // `expr_receiver_is_known_user_type` couldn't recognize the loop variable as a
        // known enum type, so a user-defined camelCase method call on it fell through to
        // the generic `map_method` fallback, which unconditionally `camel_to_snake`s the
        // method name — wrongly rewriting `s.describeMe()` to `s.describe_me()` (E0599 at
        // cargo build) even though the exact same call through a `let`/param binding of
        // that enum type was correctly preserved verbatim.
        let src = "\
enum Shape:\n    Circle(float radius)\n    Point\n\n    req string describeMe():\n        \"shape\"\n\n\
def main():\n    let c = Shape.Circle(2.0)\n    print c.describeMe()\n\n    \
let shapes = [Shape.Circle(2.0), Shape.Point]\n    for s in shapes:\n        print s.describeMe()\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(!code.contains("describe_me"),
            "camelCase enum method must not be lowered to snake_case for a for-loop variable, got:\n{}", code);
        assert!(code.matches("describeMe()").count() >= 2,
            "expected both the `let`-bound and for-loop-bound calls to preserve `describeMe()`, got:\n{}", code);
    }

    #[test]
    fn test_trait_array_literal_boxes_elements_and_borrows_in_loop() {
        // docs/book.md "Traits as types": `[Trait]` is dynamic dispatch
        // (`Vec<Box<dyn Trait>>`). Regression for the bug where the array
        // literal's own construction never boxed its elements (E0308:
        // expected `Box<dyn Trait>`, found the concrete struct) and the
        // `for` loop over it used `.iter().cloned()` (E0277: `Box<dyn
        // Trait>` isn't `Clone`) instead of borrowing.
        let src = "\
trait Drawable:\n    req string label()\n\n\
struct Circle as Drawable:\n    float r\n    req string label(): \"circle\"\n\n\
struct Square as Drawable:\n    float s\n    req string label(): \"square\"\n\n\
def main():\n    let [Drawable] shapes = [Circle(r = 1.0), Square(s = 2.0)]\n    for s in shapes:\n        print s.label()\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("Box::new(Circle"),
            "array literal element assigned into a `[Trait]` slot must be boxed, got:\n{}", code);
        assert!(code.contains("Box::new(Square"),
            "array literal element assigned into a `[Trait]` slot must be boxed, got:\n{}", code);
        assert!(!code.contains(".iter().cloned()"),
            "iterating a `Vec<Box<dyn Trait>>` must not use `.iter().cloned()` (Box<dyn Trait> isn't Clone), got:\n{}", code);
        assert!(code.contains("for s in &shapes"),
            "iterating a `Vec<Box<dyn Trait>>` should borrow (`&shapes`), got:\n{}", code);
    }

    #[test]
    fn test_trait_typed_param_is_boxed_dyn_and_call_site_boxes_argument() {
        // docs/book.md "Traits as types": a bare trait name in parameter
        // position is dynamic dispatch too (`Box<dyn Trait>`), not `impl
        // Trait` (that's the distinct `<Trait>` static-dispatch shorthand).
        // Regression for the bug where `keep(Drawable item)` transpiled
        // to `item: Drawable` (invalid Rust) / `impl Drawable` (wrong
        // dispatch kind), and the call site never boxed its argument.
        // The parameter is *consumed* (stored in a struct), so it stays by value:
        // a receiver-only param is inferred borrowed instead (next test).
        // (The trait has a `def` method, so it is not shareable — see `share_dyn_traits.rs`; the
        // all-`req` case is the next test.)
        let src = "\
trait Drawable:\n    req string label()\n    def touch()\n\n\
struct Circle as Drawable:\n    float r\n    req string label(): \"circle\"\n    def touch():\n        pass\n\n\
struct Holder:\n    Drawable d\n\n\
def keep(Drawable item):\n    let h = Holder(item)\n    print h.d.label()\n\n\
def main():\n    keep(Circle(r = 1.0))\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("item: Box<dyn Drawable>"),
            "a consumed bare trait-typed param must emit `Box<dyn Trait>`, got:\n{}", code);
        assert!(code.contains("keep(Box::new(Circle"),
            "a struct literal passed to a `Box<dyn Trait>` param must be boxed at the call site, got:\n{}", code);
    }

    #[test]
    fn all_req_trait_held_by_a_struct_is_a_shared_trait_object_and_the_struct_stays_clone() {
        // A trait with only `req` methods that some struct holds is stored as `Arc<dyn Trait>`
        // program-wide (`share_dyn_traits.rs`): every position agrees, and the holding struct (and a
        // struct holding *that*) derives `Clone` again.
        let src = "\
trait Drawable:\n    req string label()\n\n\
struct Circle as Drawable:\n    float r\n    req string label(): \"circle\"\n\n\
struct Holder:\n    Drawable? d = nil\n    [Drawable] more = []\n\n\
struct Outer:\n    [Holder] hs = []\n\n\
Holder keep(Drawable item):\n    Holder(d = item)\n\n\
def main():\n    let h = keep(Circle(r = 1.0))\n    let copy = h.clone()\n    let o = Outer(hs = [h, copy])\n    let o2 = o.clone()\n    print o2.hs.length\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("item: Arc<dyn Drawable>"), "shared trait param, got:\n{code}");
        assert!(code.contains("d: Option<Arc<dyn Drawable>>"), "shared trait optional field, got:\n{code}");
        assert!(code.contains("more: Vec<Arc<dyn Drawable>>"), "shared trait array field, got:\n{code}");
        assert!(code.contains("keep(Arc::new(Circle"), "a concrete value is wrapped at the call site, got:\n{code}");
        assert!(!code.contains("Box<dyn Drawable>"), "no `Box<dyn>` left for a shared trait, got:\n{code}");
        for name in ["Holder", "Outer"] {
            let at = code.find(&format!("pub struct {name} {{")).or_else(|| code.find(&format!("struct {name} {{"))).unwrap();
            let derive = code[..at].lines().rev().find(|l| l.starts_with("#[derive")).unwrap_or("");
            assert!(derive.contains("Clone"), "{name} must derive Clone, got `{derive}` in:\n{code}");
            assert!(!derive.contains("PartialEq"), "{name} must not derive PartialEq, got `{derive}`");
        }
    }

    #[test]
    fn test_receiver_only_trait_param_is_inferred_borrowed() {
        // Phase 2 of docs/qualifiers.md's bare-trait borrow inference: a param only used as a
        // trait-method receiver is `&dyn Trait`, and a fresh argument is borrowed unboxed.
        let src = "\
trait Drawable:\n    req string label()\n\n\
struct Circle as Drawable:\n    float r\n    req string label(): \"circle\"\n\n\
def describe(Drawable item):\n    print item.label()\n\n\
def main():\n    describe(Circle(r = 1.0))\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("item: &dyn Drawable"),
            "a receiver-only bare trait param must emit `&dyn Trait`, got:\n{}", code);
        assert!(code.contains("describe(&Circle"),
            "a fresh struct literal passed to a `&dyn Trait` param is borrowed, not boxed, got:\n{}", code);
    }

    #[test]
    fn test_trait_typed_receiver_preserves_method_casing() {
        // Regression: a receiver whose only known static type is a trait
        // name (not a concrete struct/enum) lost camelCase method-name
        // preservation, falling through to the generic camelCase->snake_case
        // fallback (`s.getLabel()` -> `s.get_label()`) instead of being kept
        // verbatim like a concrete struct/enum receiver already is.
        let src = "\
trait Drawable:\n    req string getLabel()\n\n\
struct Circle as Drawable:\n    float r\n    req string getLabel(): \"circle\"\n\n\
def main():\n    let [Drawable] shapes = [Circle(r = 1.0)]\n    for s in shapes:\n        print s.getLabel()\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("s.getLabel()"),
            "a trait-typed receiver's camelCase method call must be preserved verbatim, got:\n{}", code);
        assert!(!code.contains("s.get_label()"),
            "a trait-typed receiver's method call must not be snake_case-mangled, got:\n{}", code);
    }

    #[test]
    fn test_single_thread_uses_spawn_local() {
        // In --threading single mode, task expressions must emit tokio::task::spawn_local.
        let src = "task int work(int n):\n    return n\n\nlet t = task work(1)\n";
        let config = TranspileConfig { mode: TranspileMode::Strict, threading: ThreadingMode::Single, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("tokio::task::spawn_local"),
            "single mode: task should emit spawn_local, got:\n{}", code);
        assert!(!code.contains("tokio::spawn("),
            "single mode: should NOT emit bare tokio::spawn, got:\n{}", code);
    }

    #[test]
    fn test_single_thread_uses_local_channel() {
        // In --threading single mode, typed channel creation must use local_channel::mpsc.
        let src = "let tx, rx = channel<int>(10)\n";
        let config = TranspileConfig { mode: TranspileMode::Strict, threading: ThreadingMode::Single, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("local_channel::mpsc"),
            "single mode: channel should use local_channel::mpsc, got:\n{}", code);
    }

    #[test]
    fn test_multi_thread_uses_tokio_spawn() {
        // In --threading multi mode (default), task expressions must emit tokio::spawn.
        let src = "task int work(int n):\n    return n\n\nlet t = task work(1)\n";
        let config = TranspileConfig { mode: TranspileMode::Strict, threading: ThreadingMode::Multi, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("tokio::spawn("),
            "multi mode: task should emit tokio::spawn, got:\n{}", code);
    }

    // ── Derive-trait whitelist (`as Trait:` routed into `#[derive(...)]`) ─────────────

    #[test]
    fn struct_header_mixes_builtin_derive_and_user_trait() {
        // `Hash`/`Eq` are in the built-in KNOWN_DERIVABLE_TRAITS whitelist (not part of the
        // struct's own auto-default Debug/Clone/PartialEq) and should merge into ONE derive
        // line; `Named` is a real Boring trait and must still get its own `impl` block with
        // only its own method.
        let src = "\
trait Named:\n    req string name()\n\n\
struct Point as Hash, Eq, Named:\n    int x\n    int y\n\n    req string name():\n        \"point\"\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("#[derive(Debug, Clone, PartialEq, Eq, Hash)]"),
            "whitelisted names should merge into the struct's single derive line, got:\n{}", code);
        assert!(code.contains("impl Named for Point {"),
            "user-declared trait must still get a manual impl block, got:\n{}", code);
        assert!(!code.contains("impl Hash for Point") && !code.contains("impl Eq for Point"),
            "whitelisted derive names must NOT also get an impl block, got:\n{}", code);
    }

    #[test]
    fn trait_type_level_methods_routed_into_impl_trait_for_struct() {
        // Regression test for the book's own "Type-level methods in traits" example
        // (docs/book.md): a trait's `type def`/`type req` (no-`self`) members were only ever
        // emitted into the plain `impl Struct {}` block, never into `impl Trait for Struct {}`
        // — so `cargo build` on the generated project failed with E0046 ("not all trait items
        // implemented, missing: `create`, `type_name`") even though the transpiler itself
        // raised no error. `trait_type_method_names` (mirroring `trait_method_names` for the
        // type-level side) fixes the split; also asserts the `pub` qualifier from the Boring
        // source (illegal on a trait-impl item, E0449) is dropped only in the trait impl,
        // and kept in the plain inherent impl.
        let src = "\
trait Factory:\n    type def Self create()\n    type req string type_name()\n\n\
struct Dog as Factory:\n    string name\n\n    \
pub type def Dog create():\n        Dog(name = \"Rex\")\n\n    \
pub type req string type_name():\n        \"Dog\"\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        let trait_impl = code.split("impl Factory for Dog {").nth(1)
            .and_then(|rest| rest.split_once("\n}"))
            .map(|(body, _)| body)
            .unwrap_or_else(|| panic!("expected an `impl Factory for Dog {{ ... }}` block, got:\n{}", code));
        assert!(trait_impl.contains("fn create()"),
            "impl Factory for Dog {{}} must contain create(), got:\n{}", code);
        assert!(trait_impl.contains("fn type_name()"),
            "impl Factory for Dog {{}} must contain type_name(), got:\n{}", code);
        assert!(!trait_impl.contains("pub fn"),
            "trait-impl items must not carry `pub` (E0449), got:\n{}", code);
        // The plain inherent impl still carries `pub` (as declared in source) and still gets
        // both methods too — additive, not a replacement (Rust allows an inherent fn and a
        // trait fn of the same name on the same type to coexist).
        let inherent_impl = code.split("impl Dog {").nth(1)
            .and_then(|rest| rest.split_once("\n}"))
            .map(|(body, _)| body)
            .unwrap_or_else(|| panic!("expected a plain `impl Dog {{ ... }}` block, got:\n{}", code));
        assert!(inherent_impl.contains("pub fn create()") && inherent_impl.contains("pub fn type_name()"),
            "plain impl Dog {{}} must still carry pub create()/type_name(), got:\n{}", code);
    }

    #[test]
    fn struct_header_derive_name_extended_via_config() {
        // A project-declared name (boring.toml's `[derives]`, threaded via
        // `TranspileConfig::known_derives`) is routed exactly like a built-in one.
        let src = "struct Velocity as Component:\n    float x\n    float y\n";
        let config = TranspileConfig { known_derives: vec!["Component".to_string()], ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("#[derive(Debug, Clone, PartialEq, Component)]"),
            "project-registered derive name should merge into the derive line, got:\n{}", code);
        assert!(!code.contains("impl Component for Velocity"),
            "project-registered derive name must NOT get an impl block, got:\n{}", code);
    }

    #[test]
    fn struct_header_derive_name_not_duplicated_against_explicit_attr() {
        // Explicit `@derive(...)` and a whitelisted header name both naming `Clone` must not
        // produce two `Clone` entries (which rustc would reject as a duplicate impl).
        let src = "@derive(Debug, Clone)\nstruct Point as Clone:\n    int x\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("#[derive(Debug, Clone)]"),
            "whitelisted name already present in the explicit @derive list must not repeat, got:\n{}", code);
    }

    #[test]
    fn bare_derive_serialize_deserialize_are_qualified_and_flag_uses_serde() {
        // `@derive(Serialize, Deserialize)` (the exact bare spelling docs/book.md's `json`/
        // `fromJson` section documents) must not emit a bare, unresolvable `Deserialize` in
        // `#[derive(...)]` -- Boring never emits `use serde::{Serialize, Deserialize};`, so an
        // unqualified name there fails to compile ("cannot find derive macro"/unsatisfied
        // trait bound). Fixed by qualifying to the full path, mirroring the existing
        // `thiserror::Error` auto-qualification. `uses_serde` must also flip so Cargo.toml
        // gets the `serde` dependency even without a `json()`/`fromJson()` call in this file.
        let src = "@derive(Serialize, Deserialize)\nstruct Costume:\n    string name\n";
        let tokens = crate::lexer::lex(src).expect("lex error");
        let program = crate::parser::parse(tokens).expect("parse error");
        let out = transpile_with_config(&program, TranspileConfig::default());
        assert!(out.code.contains("#[derive(serde::Serialize, serde::Deserialize)]"),
            "bare Serialize/Deserialize should be qualified to serde::, got:\n{}", out.code);
        assert!(out.uses_serde, "a struct deriving Serialize/Deserialize must flag uses_serde");
    }

    #[test]
    fn struct_display_impl_skipped_without_debug_in_explicit_derive() {
        // The auto Display impl delegates to `{:?}` (Debug). An explicit `@derive(...)` (used
        // verbatim, unlike the no-@derive-at-all auto path, which always includes Debug) that
        // doesn't itself list Debug must not get that impl -- book.md's own fromJson example
        // (`@derive(Serialize, Deserialize)`, no Debug) hit exactly this: "`Target` doesn't
        // implement `Debug`" pointing at the generated `impl Display`.
        let src = "@derive(Serialize, Deserialize)\nstruct Costume:\n    string name\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(!code.contains("impl std::fmt::Display for Costume"),
            "no Display impl should be generated when Debug isn't in the explicit derive list, got:\n{}", code);
    }

    #[test]
    fn struct_display_impl_still_emitted_when_debug_explicit() {
        // Same shape as above, but Debug IS in the explicit list -- the impl must still be
        // generated (this is a no-regression check alongside the skip case above).
        let src = "@derive(Debug, Serialize)\nstruct Costume:\n    string name\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("impl std::fmt::Display for Costume"),
            "Display impl should still be generated when Debug is explicitly derived, got:\n{}", code);
    }

    #[test]
    fn enum_display_impl_skipped_without_debug_in_explicit_derive() {
        // The *enum* auto-Display had no equivalent of the `will_have_debug` gate the two
        // struct tests above cover, so `@derive(Clone, Serialize, Deserialize)` on an enum
        // (explicit, hence used verbatim with no auto-injected Debug) generated
        // `write!(f, "{:?}", self)` for a non-Debug type: "`Num` doesn't implement `Debug`".
        // See tests/cases/enum_derive_no_debug.br.
        let src = "@derive(Clone, Serialize, Deserialize)\nenum Num:\n    NInt(int)\n    NStr(string)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(!code.contains("impl std::fmt::Display for Num"),
            "no Display impl should be generated for an enum when Debug isn't in the explicit derive list, got:\n{}", code);
    }

    #[test]
    fn enum_display_impl_still_emitted_when_debug_explicit() {
        // No-regression companion to the skip case above: with Debug explicitly derived the
        // enum's auto-Display must still be generated (this is what lets `print someEnum`
        // and `BoringFmt<Vec<Enum>>` work at all).
        let src = "@derive(Debug, Clone, Serialize, Deserialize)\nenum Num:\n    NInt(int)\n    NStr(string)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("impl std::fmt::Display for Num"),
            "Display impl should still be generated for an enum when Debug is explicitly derived, got:\n{}", code);
    }

    #[test]
    fn enum_display_impl_still_emitted_without_any_derive_attr() {
        // No `@derive(...)` at all -> the auto path emits `#[derive(Debug, Clone, ...)]`
        // itself, so Debug is present and the auto-Display must be emitted. Guards against
        // the `will_have_debug` gate being read off the *attrs* rather than off the derive
        // names actually emitted.
        let src = "enum Num:\n    NInt(int)\n    NStr(string)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("impl std::fmt::Display for Num"),
            "an enum with no explicit @derive still auto-derives Debug, so Display must be emitted, got:\n{}", code);
    }

    #[test]
    fn field_level_attr_emitted_verbatim_above_field() {
        // `@serde(rename = "...")` (or any other attribute) directly above a field must be
        // preserved and emitted as a real Rust attribute immediately above that field --
        // needed for a JSON key that isn't a valid Boring identifier at all (e.g. "1"), which
        // a struct-wide `@serde(rename_all = "...")` can't reach. See
        // docs/json-deserialize-rename-gap.md.
        let src = "@derive(Debug)\nstruct Costume:\n    string name\n    @serde(rename = \"1\")\n    int assetIndex\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("#[serde(rename=\"1\")]\n    pub assetIndex: isize,")
            || code.contains("#[serde(rename = \"1\")]\n    pub assetIndex: isize,"),
            "field-level attr should be emitted verbatim directly above its field, got:\n{}", code);
    }

    #[test]
    fn empty_derive_attr_suppresses_auto_default_alongside_header() {
        // An explicit `@derive()` (empty args) still counts as "has_derive" — it disables
        // the struct's auto-default Debug/Clone/PartialEq entirely, while a header `as
        // Trait, ...:` list still routes its whitelisted names into the (now otherwise
        // empty) derive line. This is the intended way to combine `as Trait:` syntax with
        // an exact, non-broadened derive set (no accidental Clone/PartialEq) — see
        // docs/book.md's "derive-trait whitelist" section.
        let src = "@derive()\nstruct Paddle as Component, Debug:\n    pass\n";
        let config = TranspileConfig { known_derives: vec!["Component".to_string()], ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("#[derive(Component, Debug)]"),
            "empty @derive() should suppress the auto-default entirely, leaving only the \
             header's whitelisted names, got:\n{}", code);
        assert!(!code.contains("impl Component for Paddle"),
            "Component must still be routed to derive, not a manual impl, got:\n{}", code);
    }

    #[test]
    fn enum_header_derive_name_merges_as_extra_derive_line() {
        // A non-unit enum's auto-default is only Debug/Clone/PartialEq (unlike the
        // unit-only-variants case, which already includes Hash/Eq/Copy) — `Hash` here is
        // genuinely new and must appear in its own extra `#[derive(...)]` line.
        let src = "enum Shape as Hash:\n    Circle(int)\n    Square(int, int)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("#[derive(Debug, Clone, PartialEq)]"),
            "existing auto-derive line should be untouched, got:\n{}", code);
        assert!(code.contains("#[derive(Hash)]"),
            "whitelisted enum header name not already covered by the auto-derive line \
             should get its own extra #[derive(...)] line, got:\n{}", code);
    }

    #[test]
    fn ext_block_rejects_whitelisted_derive_name() {
        // Rust can't retroactively attach #[derive(...)] to a type defined elsewhere — an
        // `ext ... as <whitelisted>:` must be a hard transpile error, not silently emit
        // `impl <Trait> for X { <every ext method> }`.
        let src = "\
struct Foo:\n    int x\n\n\
ext Foo as Debug:\n    req int double():\n        self.x * 2\n";
        let tokens = crate::lexer::lex(src).expect("lex error");
        let program = crate::parser::parse(tokens).expect("parse error");
        let out = transpile_with_config(&program, TranspileConfig::default());
        assert!(!out.errors.is_empty(), "expected a transpile error for 'ext Foo as Debug:'");
        assert!(out.errors.iter().any(|e| e.message.contains("derive macro")),
            "error should explain Debug is a derive macro, got: {:?}", out.errors);
        assert!(!out.code.contains("impl Debug for Foo"),
            "must not emit an impl block for a derive-macro name via ext, got:\n{}", out.code);
    }

    // ── `boring.toml [external_fns]` / `KNOWN_EXTERNAL_FN_BORROWS` ──────────────────────
    // `mem_borrow_builtins` (tests/cases/mem_borrow_builtins.br, tests/transpile.rs) already
    // covers the built-in table end-to-end through a real `cargo run`. The three unit tests
    // below cover project-declared `TranspileConfig::external_fns` entries at the transpiler
    // level (no real external crate needed, since these only check the generated Rust
    // *string* — a full compile+run of a project-declared entry against real crates like
    // `zip`/`resvg` was done separately, outside this repo's own test suite).

    #[test]
    fn external_fns_static_type_call_applies_project_declared_borrows() {
        // `Type.method(args)` — the "static call on an unregistered external type" shape
        // (`try_emit_type_method_call`'s own external fallback), the third and, until this
        // fix, unwired call site (found via the `Tree.from_str(text, opt)` real-crate repro
        // — `resvg.render(...)`, a free function, already worked; this one silently didn't).
        let src = "def main():\n    var int a = 1\n    var int b = 2\n    Widget.build(a, b)\n";
        let config = TranspileConfig {
            external_fns: vec![("Widget".to_string(), "build".to_string(), vec!["".to_string(), "&mut".to_string()])],
            ..Default::default()
        };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("Widget::build(a, &mut b)"),
            "static Type.method(args) call should apply project-declared arg borrows, got:\n{}", code);
    }

    #[test]
    fn external_fns_method_name_is_normalized_to_snake_case_before_matching() {
        // Regression test: `boring.toml [external_fns]` stores a key's method half verbatim
        // from the TOML text (e.g. `"buildThing"`, matching the `.br` source's own camelCase
        // call spelling) — never run through `camel_to_snake` at parse time. Every call site
        // passes the *already*-`camel_to_snake`'d Rust method name to
        // `external_fn_arg_borrows`, so the lookup itself must normalize the stored entry
        // too, or a project's own whitelist entry never matches at all. Never caught by
        // `mem.swap`/`replace`/`take` (identical Boring-source/Rust spellings already) —
        // only surfaced via the `ZipFile::readToEnd` (→ `read_to_end`) real-crate repro.
        let src = "def main():\n    var int a = 1\n    var int b = 2\n    Widget.buildThing(a, b)\n";
        let config = TranspileConfig {
            external_fns: vec![("Widget".to_string(), "buildThing".to_string(), vec!["".to_string(), "&mut".to_string()])],
            ..Default::default()
        };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("Widget::build_thing(a, &mut b)"),
            "camelCase TOML method name should still match the snake_case Rust call site, got:\n{}", code);
    }

    #[test]
    fn external_fns_preserves_positional_by_value_placeholder_mid_array() {
        // Regression test for the `resvg::render` real-crate repro's other bug:
        // `["&", "", "&mut"]`'s middle by-value placeholder must stay at index 1, not be
        // dropped and shift the trailing `"&mut"` left (`BoringToml::extract_borrow_array`,
        // src/main.rs — this test exercises the *consuming* end of that fix, the TOML
        // parsing side has its own dedicated unit tests in main.rs).
        let src = "def main():\n    var int a = 1\n    var int b = 2\n    var int c = 3\n    Lib.call3(a, b, c)\n";
        let config = TranspileConfig {
            external_fns: vec![("Lib".to_string(), "call3".to_string(),
                vec!["&".to_string(), "".to_string(), "&mut".to_string()])],
            ..Default::default()
        };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("Lib::call3(&a, b, &mut c)"),
            "middle by-value arg must stay unprefixed, first/last keep their own borrows, got:\n{}", code);
    }

    // ── `EXTERNAL_MUT_BUFFER_METHODS` / `EXTERNAL_NON_CLONE_TYPES` ───────────────────────
    // A mutable array binding passed to a `std::io::Read`-family method is lent as `&mut`;
    // a non-`Clone` std type passed by value to a Boring function is moved, not `.clone()`d.
    // `tests/cases/external_read_buffer.br` covers the same through a real `cargo run`.

    #[test]
    fn external_read_var_only_param_array_is_rejected() {
        // `var` alone is rebindable, not content-mutable: it cannot be a read buffer.
        let src = "use std.net.TcpStream\nuse std.io.Read\n\ndef f(mut TcpStream sock, var [uint8] b) throws:\n    let n = try? sock.read(b)\n    guard let n else throw \"x\"\n";
        let errs = transpile_src_errors(src);
        assert!(errs.contains("cannot pass `b` as a mutable buffer"),
            "bare `var` array param must be rejected, got: {}", errs);
    }

    #[test]
    fn external_read_var_only_local_array_is_rejected() {
        let src = "use std.net.TcpStream\nuse std.io.Read\n\ndef f(mut TcpStream sock) throws:\n    var [uint8] buf = [0 as uint8 for i in 0..<64]\n    let n = try? sock.read(buf)\n    guard let n else throw \"x\"\n";
        let errs = transpile_src_errors(src);
        assert!(errs.contains("cannot pass `buf` as a mutable buffer"),
            "bare `var` array local must be rejected, got: {}", errs);
    }

    #[test]
    fn external_read_mut_param_array_is_reborrowed() {
        // `mut [T]` (content-mutable) is the idiomatic buffer parameter; `var` also works.
        let src = "use std.net.TcpStream\nuse std.io.Read\n\ndef int f(mut TcpStream sock, mut [uint8] b) throws:\n    let n = try? sock.read(b)\n    guard let n else throw \"x\"\n    return n as int\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("sock.read(&mut *b)"),
            "`mut [uint8]` param must be reborrowed (`&mut *b`), not `&mut b`, got:\n{}", code);
    }

    #[test]
    fn external_read_var_local_array_is_borrowed_mutably() {
        let src = "use std.net.TcpStream\nuse std.io.Read\n\ndef int f(mut TcpStream sock) throws:\n    var mut [uint8] buf = [0 as uint8 for i in 0..<64]\n    let n = try? sock.read_exact(buf)\n    guard let n else throw \"x\"\n    return 1\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("sock.read_exact(&mut buf)"),
            "`var mut [uint8]` local must be passed as `&mut buf`, got:\n{}", code);
    }

    // ── `mut [T]` / `mut {K=V}` / `mut {T}` parameters lend the caller's content as `&mut` ──────
    // (docs/book.md "`mut` vs `var` on a struct parameter"; tests/cases/mut_collection_param.br
    // covers the same through a real `cargo run` and `boring run`.)

    // ── a bare-`mut` local collection aliasing another variable keeps its dict/set-ness ────────
    // `mut {K=V} x = other` parses to `Type::Mut(Dict(..))`; `emit_let` must look through the
    // wrapper (`decl_ty`) so `dict_vars`/`set_vars`/`vec_vars` are populated exactly as for `var`.

    #[test]
    fn mut_dict_local_from_var_index_assigns_via_insert() {
        for kw in ["mut", "var"] {
            let src = format!("def int render({{string=int}} env):\n    {kw} {{string=int}} loop_env = env\n    loop_env[\"k\"] = 1\n    loop_env.length\n\ndef main():\n    print render({{\"a\" = 0}})\n");
            let code = transpile_src_with_config(&src, TranspileConfig::default());
            assert!(code.contains("loop_env.insert(Arc::<str>::from(\"k\"), 1);"), "`{}` dict must insert, got:\n{}", kw, code);
            assert!(!code.contains("loop_env[\"k\"] = 1"), "`{}` must not array-index, got:\n{}", kw, code);
        }
    }

    #[test]
    fn mut_array_local_from_var_index_assigns_as_vec() {
        let src = "def int f([int] a):\n    mut [int] b = a\n    b[0] = 5\n    b.length\n\ndef main():\n    print f([1, 2])\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("b[(0) as usize] = 5;"), "got:\n{}", code);
    }

    // `mut [float] e` is stored as `Type::Mut(Array(float))` in `var_types`; the float-element
    // lookups (`float_array_elem_ty` for `.sum::<T>()`, `infer_float_width_inner`'s index arm,
    // `let x = arr[i]` tracking, `emit_for`'s element type) must look through the `Mut` wrapper.
    #[test]
    fn mut_float_array_local_sum_uses_float_turbofish() {
        for (kw, elem, rust) in [("mut", "float", "f64"), ("var", "float", "f64"), ("mut", "float32", "f32"), ("var", "float32", "f32")] {
            let src = format!("def {elem} f([{elem}] x):\n    {kw} [{elem}] e = x |> map((v): v)\n    let {elem} s = e |> sum()\n    s\n\ndef main():\n    print f([1.0, 2.0])\n");
            let code = transpile_src_with_config(&src, TranspileConfig::default());
            assert!(code.contains(&format!("sum::<{rust}>()")), "`{kw} [{elem}]` sum must use {rust}, got:\n{code}");
            assert!(!code.contains("sum::<i64>()"), "`{kw} [{elem}]` must not default to i64, got:\n{code}");
        }
    }

    // `push` onto a `mut`/`var mut` local `[Trait]` array must box the element:
    // `var_types` keeps the `Type::Mut(..)` wrapper, which the push path's element-trait lookup
    // matched `Type::Array` against directly (was `a.push(Text { .. })`, rustc E0308).
    #[test]
    fn mut_trait_array_local_push_boxes_element() {
        for kw in ["mut", "var mut"] {
            let src = format!("trait Widget:\n    req string render()\n\nstruct Text as Widget:\n    pub string text\n    req string render(): text\n\ndef main():\n    {kw} [Widget] a = []\n    a.push(Text(\"x\"))\n    print a.length\n");
            let code = transpile_src_with_config(&src, TranspileConfig::default());
            assert!(code.contains("a.push(Box::new(Text"), "`{kw} [Widget]` push must box the element, got:\n{code}");
        }
    }

    // The element-`mut` spelling `[mut Trait]` (`Type::Array(Type::Mut(Named))`) must be treated as
    // a `[Trait]` everywhere: literal elements boxed, `push` boxed, `for` borrows instead of
    // `.iter().cloned()` (`Box<dyn Trait>` is not `Clone`).
    #[test]
    fn mut_elem_trait_array_local_boxes_literal_and_push() {
        let src = "trait Poker:\n    req tap(int x)\n\nstruct Sq as Poker:\n    var int n = 0\n    req tap(int x):\n        print x\n\ndef main():\n    mut [mut Poker] mps = [Sq(), Sq()]\n    mps.push(Sq())\n    for p in mps:\n        p.tap(1)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(
            code.contains("vec![Box::new(Sq") && code.matches("Box::new(Sq").count() >= 3,
            "literal elements must be boxed, got:\n{code}"
        );
        assert!(code.contains("mps.push(Box::new(Sq"), "push must box, got:\n{code}");
        assert!(code.contains("for p in &mps") && !code.contains("mps.iter().cloned()"), "for must borrow, got:\n{code}");
    }

    // A `mut`/`var mut` collection local passed by value to a constructor / enum variant must be
    // cloned like its `var`/`let` twin (was a move -> E0382 on reuse): `var_types` keeps the
    // `Type::Mut(..)` wrapper, which `emit_expr_owned`'s clone arm didn't look through. Array/set
    // locals of any spelling are cloned in a constructor field as well.
    #[test]
    fn mut_collection_local_passed_by_value_is_cloned_like_var() {
        for (kw, ty, init) in [
            ("var", "{string=int}", "{=}"), ("mut", "{string=int}", "{=}"), ("var mut", "{string=int}", "{=}"),
            ("var", "[int]", "[1]"), ("mut", "[int]", "[1]"), ("var mut", "[int]", "[1]"),
            ("mut", "{int}", "{1}"), ("var mut", "{int}", "{1}"),
        ] {
            let src = format!("struct Holder:\n    pub var {ty} f\n\ndef main():\n    {kw} {ty} x = {init}\n    let a = Holder(x)\n    let b = Holder(x)\n    print a.f.length + b.f.length + x.length\n");
            let code = transpile_src_with_config(&src, TranspileConfig::default());
            assert_eq!(code.matches("Holder { f: x.clone() }").count(), 2,
                "`{kw} {ty}` by-value ctor arg must be cloned both times, got:\n{code}");
            assert!(!code.contains("Holder { f: x }"), "`{kw} {ty}` must not be moved, got:\n{code}");
        }
        let src = "enum W:\n    L([int] items)\n\ndef main():\n    var mut [int] x = [1]\n    let a = W.L(x)\n    let b = W.L(x)\n    print x.length\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert_eq!(code.matches("W::L(x.clone())").count(), 2, "enum variant arg must be cloned, got:\n{code}");
    }

    // A call that lends `&mut v` and also reads `v` by value in another argument must evaluate
    // the by-value arguments first: a plain `&mut v` argument has no two-phase borrow (E0502).
    #[test]
    fn lent_collection_local_also_read_by_value_hoists_other_args() {
        let src = "struct Holder:\n    pub [int] xs\n\nstruct Bag:\n    pub int n\n\n    def int grow_with(mut [int] a, Holder h):\n        a.push(9)\n        a.length + h.xs.length\n\ndef int grow_with(mut [int] a, Holder h):\n    a.push(9)\n    a.length + h.xs.length\n\ndef int other(mut [int] a, [int] b):\n    a.push(1)\n    b.length\n\ndef main():\n    mut [int] v = [1, 2, 3]\n    mut [int] w = [4]\n    mut Bag b = Bag(1)\n    print grow_with(v, Holder(v))\n    print b.grow_with(w, Holder(w))\n    print other(v, w)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("({ let __arg1 = &Holder { xs: v.clone() }; grow_with(&mut v, __arg1) })"),
            "free-function call must hoist the by-value argument, got:\n{code}");
        assert!(code.contains("({ let __arg1 = &Holder { xs: w.clone() }; b.grow_with(&mut w, __arg1) })"),
            "method call must hoist the by-value argument, got:\n{code}");
        // No overlap -> the call is left exactly as before.
        assert!(code.contains("other(&mut v, &w)"), "non-overlapping call must stay untouched, got:\n{code}");
        // A longer identifier / another receiver's field is not a mention of the lent place.
        let mut args = vec!["&mut v".to_string(), "Holder { xs: vv.clone(), ys: h.v }".to_string()];
        assert_eq!(Transpiler::hoist_lent_conflicts(&mut args), "");
        assert_eq!(args[1], "Holder { xs: vv.clone(), ys: h.v }");
    }

    // Same hazard through the dedicated lock-wrapper dispatchers ('actor / 'guard / 'observed local,
    // `self.field` of an 'actor type): the by-value arguments are hoisted into temporaries *before*
    // the lock is taken, and the whole call is wrapped in a block.
    #[test]
    fn lent_collection_local_also_read_by_value_hoists_through_lock_wrappers() {
        let src = "struct Holder:\n    pub [int] xs\n\nstruct Bag:\n    pub int n\n\n    def int grow_with(mut [int] a, Holder h):\n        a.push(9)\n        a.length + h.xs.length\n\nstruct Wrap:\n    mut Bag'actor inner = Bag(1)\n\n    def int via_self(mut [int] a):\n        self.inner.grow_with(a, Holder(a))\n\ndef main():\n    mut [int] v = [1]\n    var mut Bag'actor x = Bag(1)\n    print x.grow_with(v, Holder(v))\n    mut [int] w = [2]\n    var mut Bag'guard y = Bag(1)\n    print y.grow_with(w, Holder(w))\n    mut [int] u = [3]\n    var mut Bag'actor'observed z = Bag(1)\n    print z.grow_with(u, Holder(u))\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        for (lock, v) in [("x.lock().unwrap()", "v"), ("y.write().unwrap()", "w"), ("z.value.lock().unwrap()", "u")] {
            let want = format!("({{ let __arg1 = &Holder {{ xs: {v}.clone() }}; {lock}.grow_with(&mut {v}, __arg1) }})");
            assert!(code.contains(&want), "lock-wrapper call must hoist the by-value argument ({want}), got:\n{code}");
        }
        assert!(code.contains("({ let __arg1 = &Holder { xs: a.clone() }; self.inner.lock().unwrap().grow_with(&mut *a, __arg1) })"),
            "`self.field'actor` call must hoist the by-value argument, got:\n{code}");
    }

    // An argument that reads an `'actor` field takes a `MutexGuard` that is a temporary of the
    // enclosing call statement; a callee (or locked receiver) that locks the same value again
    // self-deadlocks. Free-function call: the argument is scoped in its own block. Call through a
    // locked receiver: the arguments are hoisted before the receiver is locked.
    #[test]
    fn actor_field_argument_releases_its_guard_before_the_call() {
        let src = "struct Env:\n    var int depth\n\nstruct Interp:\n    var Env'actor current_env\n    var int hits\n\n    def int bump(int k, int o):\n        hits += k\n        hits + o\n\ndef int helper(mut Interp'actor interp, int k, Env'actor env):\n    interp.hits += k\n    interp.hits + env.depth\n\ndef int run(mut Interp'actor interp):\n    helper(interp, interp.hits, interp.current_env)\n\ndef int via_method(mut Interp'actor interp):\n    interp.bump(interp.hits, 1)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("helper(&interp, { let __a = interp.lock().unwrap().hits"),
            "scalar field argument must scope its guard, got:\n{code}");
        assert!(code.contains("&{ let __a = Arc::clone(&interp.lock().unwrap().current_env); __a })"),
            "actor field argument must scope its guard, got:\n{code}");
        assert!(code.contains("({ let __arg0 = { let __a = interp.lock().unwrap().hits"),
            "locked-receiver call must hoist the lock-taking argument before the receiver lock, got:\n{code}");
        assert!(code.contains("; interp.lock().unwrap().bump(__arg0, __arg1) })"),
            "locked-receiver call must use the hoisted temporaries, got:\n{code}");
    }

    // A collection field of an `'actor` value passed to a by-reference `[T]` parameter used to be
    // emitted as a borrow *into* the lock guard (`&x.lock().unwrap().log`), holding the guard for
    // the whole call. When the field type is provably `Clone` it is cloned out of the guard inside
    // its own block; when it is not (`[Trait]` -> `Vec<Box<dyn Trait>>`, a struct without `Clone`)
    // or the parameter is `mut`, the borrow is kept.
    #[test]
    fn actor_collection_field_by_ref_argument_is_cloned_out_of_the_guard_when_clone() {
        let src = "trait Sized:\n    req int size()\n\nstruct Sq as Sized:\n    var int n\n\n    req int size():\n        n\n\ntrait Bump:\n    def bump()\n\nstruct Bp as Bump:\n    var int n\n\n    def bump():\n        n += 1\n\n@derive(Debug)\nstruct Raw:\n    var int n\n\nstruct Interp:\n    var int hits\n    var [int] log\n    var [string] words\n    var {string=int} table\n    var [[int]] grid\n    var [Sized] shapes\n    var [Bump] bumps\n    var [Raw] raws\n\ndef int h_arr(mut Interp'actor interp, [int] v):\n    interp.hits += v.length\n    interp.hits\n\ndef int h_words(mut Interp'actor interp, [string] v):\n    interp.hits += v.length\n    interp.hits\n\ndef int h_table(mut Interp'actor interp, {string=int} v):\n    interp.hits += v.length\n    interp.hits\n\ndef int h_grid(mut Interp'actor interp, [[int]] v):\n    interp.hits += v.length\n    interp.hits\n\ndef int h_shapes(Interp'actor interp, [Sized] v):\n    v.length\n\ndef int h_bumps(Interp'actor interp, [Bump] v):\n    v.length\n\ndef int h_raws(Interp'actor interp, [Raw] v):\n    v.length\n\ndef int h_mut(mut Interp'actor interp, mut [int] v):\n    v.push(1)\n    v.length\n\ndef int t(mut Interp'actor interp):\n    h_arr(interp, interp.log) + h_words(interp, interp.words) + h_table(interp, interp.table) + h_grid(interp, interp.grid) + h_shapes(interp, interp.shapes) + h_bumps(interp, interp.bumps) + h_raws(interp, interp.raws) + h_mut(interp, interp.log)\n\ndef main():\n    let interp'actor = Interp(hits = 1, log = [1], words = [\"a\"], table = {\"k\" = 1}, grid = [[1]], shapes = [Sq(n = 1)], bumps = [Bp(n = 1)], raws = [Raw(n = 2)])\n    print t(interp)\n";
        for (threading, tok) in [
            (ThreadingMode::Multi, "interp.lock().unwrap()"),
            (ThreadingMode::Single, "interp.borrow()"),
        ] {
            let code = transpile_src_with_config(src, TranspileConfig { threading, ..TranspileConfig::default() });
            for (callee, field) in [("h_arr", "log"), ("h_words", "words"), ("h_table", "table"), ("h_grid", "grid")] {
                let want = format!("{callee}(&interp, &{{ let __a = {tok}.{field}.clone(); __a }})");
                assert!(code.contains(&want), "`Clone` field must be cloned out of the guard ({want}), got:\n{code}");
            }
            // `[Trait]` of an all-`req` trait is `Vec<Arc<dyn Trait>>` (`share_dyn_traits.rs`): `Clone`.
            let want = format!("h_shapes(&interp, &{{ let __a = {tok}.shapes.clone(); __a }})");
            assert!(code.contains(&want), "shared `[Trait]` field must be cloned out of the guard ({want}), got:\n{code}");
            // Not provably `Clone`: the borrow into the guard is kept.
            assert!(code.contains(&format!("h_bumps(&interp, &{tok}.bumps)")),
                "`[Trait]` (`Vec<Box<dyn Trait>>`, `def` method) field must keep its borrow, got:\n{code}");
            assert!(code.contains(&format!("h_raws(&interp, &{tok}.raws)")),
                "field of a struct without `Clone` must keep its borrow, got:\n{code}");
            // A `mut` borrow must reach the real storage: never cloned.
            assert!(!code.contains("h_mut(&interp, &{ let __a"), "`mut` borrow must not be cloned, got:\n{code}");
        }
    }

    // A collection field of an `'actor` value passed BY VALUE to a method called through the
    // locked receiver was emitted `{ let __a = x.lock().unwrap().log; __a }` — a move out of the
    // `MutexGuard` (E0507). The by-value read now clones, inside the guard-releasing block.
    #[test]
    fn actor_collection_field_by_value_method_argument_is_cloned_out_of_the_guard() {
        let src = "struct Interp:\n    var int hits\n    var [int] log\n    var {string=int} table\n\n    def int bump_all([int] vs, {string=int} t):\n        hits += vs.length + t.length\n        hits\n\ndef int t(mut Interp'actor interp):\n    interp.bump_all(interp.log, interp.table)\n";
        for (threading, tok) in [
            (ThreadingMode::Multi, "interp.lock().unwrap()"),
            (ThreadingMode::Single, "interp.borrow()"),
        ] {
            let code = transpile_src_with_config(src, TranspileConfig { threading, ..TranspileConfig::default() });
            for (i, field) in [(0, "log"), (1, "table")] {
                let want = format!("let __arg{i} = &{{ let __a = {tok}.{field}.clone(); __a }};");
                assert!(code.contains(&want), "by-value field read must clone ({want}), got:\n{code}");
            }
        }
    }

    #[test]
    fn release_guard_temps_only_wraps_owned_lock_taking_arguments() {
        let mut args = vec![
            "&interp".to_string(),                                                      // plain place: untouched
            "&Arc::clone(&x.lock().unwrap().env)".to_string(),                           // owned clone: wrapped, `&` outside
            "x.lock().unwrap().hits".to_string(),                                       // owned scalar: wrapped
            "&x.lock().unwrap().log".to_string(),                                       // reference into the guard: untouched
            "&mut x.lock().unwrap().log".to_string(),                                   // lend: untouched
            "|v| v + x.lock().unwrap().hits".to_string(),                               // closure: untouched
            "7".to_string(),
        ];
        Transpiler::release_guard_temps(&mut args);
        assert_eq!(args, vec![
            "&interp",
            "&{ let __a = Arc::clone(&x.lock().unwrap().env); __a }",
            "{ let __a = x.lock().unwrap().hits; __a }",
            "&x.lock().unwrap().log",
            "&mut x.lock().unwrap().log",
            "|v| v + x.lock().unwrap().hits",
            "7",
        ]);
        // A locked-receiver call must not borrow into a second guard of the same lock (self-deadlock):
        // the place is snapshotted inside a guard-releasing block, then hoisted like any owned argument.
        let mut args = vec!["{ let __a = x.lock().unwrap().n; __a }".to_string(), "&y.lock().unwrap().log".to_string()];
        let prelude = Transpiler::hoist_lent_conflicts_locked(&mut args);
        assert_eq!(prelude, "let __arg0 = { let __a = x.lock().unwrap().n; __a }; let __arg1 = &{ let __a = y.lock().unwrap().log.clone(); __a }; ");
        assert_eq!(args, vec!["__arg0", "__arg1"]);
    }

    #[test]
    fn struct_field_read_by_value_is_cloned_not_partially_moved() {
        let src = "struct Holder:\n    pub var [int] xs\n    pub var int n\n\nstruct Wrap:\n    pub var Holder inner\n\nenum Wrapper:\n    List([int] items)\n    Boxed(Holder h)\n    Num(int n)\n\ndef int len_of(Holder h):\n    h.xs.length\n\ndef int f():\n    let h = Holder([4, 5, 6], 1)\n    let w = Wrapper.List(h.xs)\n    let k = h.n\n    let nn = Wrapper.Num(h.n)\n    let wr = Wrap(h)\n    let b = Wrapper.Boxed(wr.inner)\n    len_of(h) + k + len_of(wr.inner)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("Wrapper::List(h.xs.clone())"), "collection field must be cloned, got:\n{code}");
        assert!(code.contains("Wrapper::Boxed(wr.inner.clone())"), "struct field must be cloned, got:\n{code}");
        assert!(code.contains("len_of(&wr.inner)"), "bare struct param is auto-borrowed (no clone), got:\n{code}");
        assert!(code.contains("Wrapper::Num(h.n)"), "Copy scalar field must stay a plain read, got:\n{code}");
    }

    #[test]
    fn if_let_over_inferred_optional_method_result_clones_by_value_arg() {
        // `opt` has no annotation: its `Decl?` type comes from `find`'s declared return type.
        let src = "struct Decl:\n    pub var [int] ps\n\nstruct Reg:\n    pub var int n\n\n    req Decl? find(int k):\n        nil\n\ndef int run_body(Decl d):\n    d.ps.length\n\ndef int f(Reg r):\n    let opt = r.find(1)\n    if let d = opt:\n        let a = run_body(d)\n        return a + d.ps.length\n    0\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("run_body(&d)"), "if-let binding passed to an auto-borrowed param needs no clone, got:\n{code}");
    }

    #[test]
    fn optional_method_result_shape_does_not_leak_into_same_named_rebind() {
        // `result` is first a `Decl?` (method call), then rebound to a plain `Decl` by a free
        // call: the later `match result:` must not be emitted with `Some(..)` arms.
        let src = "enum Decl:\n    A(int n)\n    B\n\nstruct Reg:\n    pub var int n\n\n    req Decl? find(int k):\n        nil\n\ndef Decl mk():\n    Decl.B\n\ndef int f(Reg r):\n    let result = r.find(1)\n    if let d = result:\n        return 1\n    let result = mk()\n    match result:\n        A(n): n\n        B: 0\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(!code.contains("Some(Decl::A"), "stale Optional shape leaked into the rebind, got:\n{code}");
    }

    #[test]
    fn field_of_indexed_struct_local_passed_to_actor_method_is_cloned() {
        // `p = ps[i]` is only known through `ps`'s element type; `env` is an `'actor` receiver,
        // whose method-call arguments take a different emission path than a plain receiver.
        let src = "struct Param:\n    pub var string name\n\nstruct Env:\n    pub var int n\n\n    req int get(string name):\n        name.length\n\ndef int f([Param] ps, Env'actor env):\n    let p = ps[0]\n    let a = env.get(p.name)\n    a + p.name.length\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains(".get(p.name.clone())"), "field read passed by value to an actor method must be cloned, got:\n{code}");
    }

    // Last-use analysis (`last_use.rs`): the last by-value read of an owned local is a move,
    // earlier ones still clone; a variable read again afterwards keeps every read a clone.
    fn last_use_code(body: &str) -> String {
        let src = format!("struct Holder:\n    pub var [int] xs\n\ndef int len_of(Holder h):\n    h.xs.length\n\ndef int both([int] a, Holder h):\n    a.length\n\n{body}");
        transpile_src_with_config(&src, TranspileConfig::default())
    }

    #[test]
    fn last_use_of_owned_local_is_moved_earlier_use_is_cloned() {
        let code = last_use_code("def main():\n    mut [int] v = [1]\n    let a = Holder(v)\n    let b = Holder(v)\n    print len_of(a) + len_of(b)\n");
        assert!(code.contains("let a: Holder = Holder { xs: v.clone() };"), "earlier use must clone, got:\n{code}");
        assert!(code.contains("let b: Holder = Holder { xs: v };"), "last use must move, got:\n{code}");
    }

    #[test]
    fn last_use_move_applies_to_return_of_a_local_struct() {
        let code = last_use_code("def Holder mk([int] s):\n    let h = Holder(s)\n    return h\n\ndef main():\n    print len_of(mk([1]))\n");
        assert!(code.contains("return h;") && !code.contains("return h.clone();"), "got:\n{code}");
    }

    #[test]
    fn last_use_keeps_clone_when_read_again_later() {
        let code = last_use_code("def main():\n    mut [int] v = [1]\n    let a = Holder(v)\n    print v.length + len_of(a)\n");
        assert!(code.contains("Holder { xs: v.clone() }"), "later read must keep the clone, got:\n{code}");
    }

    #[test]
    fn last_use_keeps_clone_inside_loop_over_outer_binding() {
        for head in ["while i < 2:", "for i in 0..<2:", "loop:"] {
            let body = format!("def main():\n    mut [int] v = [1]\n    var int i = 0\n    {head}\n        let a = Holder(v)\n        i += 1\n        if i > 1:\n            break\n");
            let code = last_use_code(&body);
            assert!(code.contains("Holder { xs: v.clone() }"), "`{head}` must keep the clone, got:\n{code}");
        }
    }

    #[test]
    fn last_use_inside_a_loop_is_a_move_when_the_binding_is_declared_in_it() {
        let code = last_use_code("def main():\n    for i in 0..<2:\n        mut [int] v = [i]\n        let a = Holder(v)\n        print len_of(a)\n");
        assert!(code.contains("Holder { xs: v }"), "per-iteration binding may be moved, got:\n{code}");
    }

    #[test]
    fn last_use_keeps_clone_for_defer_closure_and_same_statement_and_params() {
        // later defer read
        let code = last_use_code("def main():\n    mut [int] v = [1]\n    defer:\n        print v.length\n    let a = Holder(v)\n    print len_of(a)\n");
        assert!(code.contains("Holder { xs: v.clone() }"), "defer: got:\n{code}");
        // later closure capture
        let code = last_use_code("def main():\n    mut [int] v = [1]\n    let a = Holder(v)\n    let f = (n): v.length + n\n    print len_of(a) + f(0)\n");
        assert!(code.contains("Holder { xs: v.clone() }"), "closure: got:\n{code}");
        // borrowed in the same call (`&v`) — a move would be E0505
        let code = last_use_code("def main():\n    mut [int] v = [1]\n    print both(v, Holder(v))\n");
        assert!(code.contains("Holder { xs: v.clone() }"), "same statement: got:\n{code}");
        // a (borrowed) parameter is never moved
        let code = last_use_code("def Holder mk([int] s):\n    return Holder(s)\n\ndef main():\n    print len_of(mk([1]))\n");
        assert!(code.contains("Holder { xs: s.clone() }"), "param: got:\n{code}");
    }

    #[test]
    fn last_use_keeps_clone_when_name_is_also_pattern_or_for_bound() {
        let code = last_use_code("def main():\n    mut [int] v = [1]\n    for v in [[1], [2]]:\n        print v.length\n    let a = Holder(v)\n    print len_of(a)\n");
        assert!(code.contains("Holder { xs: v.clone() }"), "for-bound name must stay tainted, got:\n{code}");
    }

    #[test]
    fn last_use_moves_match_pattern_binding_and_owned_param() {
        let code = last_use_code("enum Payload:\n    A([int])\n    B(string)\n\ndef int consume([int]'inline xs):\n    let h = Holder(xs)\n    return h.xs.length\n\ndef int process_owned(Payload'inline p):\n    match p:\n        Payload.A(items):\n            return consume(items)\n        Payload.B(s):\n            return s.length\n\ndef int forward_owned([int]'inline xs):\n    return consume(xs)\n\ndef main():\n    print process_owned(Payload.A([1, 2]))\n    print forward_owned([1, 2])\n");
        assert!(code.contains("return consume(items);"), "owned match pattern binding must move, got:\n{code}");
        assert!(code.contains("return consume(xs);"), "owned param must move, got:\n{code}");
        assert!(!code.contains("items.clone()") && !code.contains("xs.clone()"), "unexpected clone, got:\n{code}");
    }

    #[test]
    fn last_use_keeps_clone_for_borrowed_match_subject() {
        let code = last_use_code("enum Payload:\n    A([int])\n    B(string)\n\ndef int consume([int]'inline xs):\n    let h = Holder(xs)\n    return h.xs.length\n\ndef int process_borrowed(Payload& p):\n    match p:\n        Payload.A(items):\n            return consume(items)\n        Payload.B(s):\n            return s.length\n\ndef main():\n    print process_borrowed(Payload.A([1, 2]))\n");
        assert!(code.contains("return consume(items.clone());"), "borrowed match subject must clone, got:\n{code}");
    }

    #[test]
    fn untyped_literal_collection_local_is_cloned_when_reused() {
        let code = last_use_code("enum W:\n    L([int] items)\n\ndef main():\n    let b = [1, 2]\n    let w = W.L(b)\n    print b.length\n");
        assert!(code.contains("W::L(b.clone())"), "unannotated literal local must clone, got:\n{code}");
    }

    // The by-value clone must not touch lent `mut [T]` params / `var` out-params at call sites.
    #[test]
    fn by_value_clone_keeps_lent_mut_param_and_var_out_param_by_reference() {
        let src = "def grow(mut [int] xs):\n    xs.push(1)\n\ndef refill(var [int] o):\n    o = [7]\n\ndef main():\n    var mut [int] v = [1]\n    grow(v)\n    refill(v)\n    print v.length\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("grow(&mut v)"), "lent param must stay `&mut`, got:\n{code}");
        assert!(code.contains("refill(&mut v)"), "out-param must stay `&mut`, got:\n{code}");
        assert!(!code.contains("grow(v.clone())") && !code.contains("refill(v.clone())"), "got:\n{code}");
    }

    #[test]
    fn mut_set_local_from_var_add_is_insert() {
        for kw in ["mut", "var"] {
            let src = format!("def int f({{int}} s):\n    {kw} {{int}} t = s\n    t.add(9)\n    t.length\n\ndef main():\n    print f({{1}})\n");
            let code = transpile_src_with_config(&src, TranspileConfig::default());
            assert!(code.contains("t.insert(9);"), "`{}` set add must be insert, got:\n{}", kw, code);
        }
    }

    #[test]
    fn mut_array_param_index_assign_is_lent_not_cloned() {
        let src = "def fill(mut [uint8] b):\n    b[0] = 7 as uint8\n\ndef main():\n    mut [uint8] buf = [0 as uint8 for i in 0..<4]\n    fill(buf)\n    print buf[0]\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn fill(b: &mut Vec<u8>)"), "signature must be `&mut Vec<u8>`, got:\n{}", code);
        assert!(code.contains("fill(&mut buf);"), "call site must lend `&mut buf`, got:\n{}", code);
        assert!(!code.contains("fill(buf.clone())"), "must not clone, got:\n{}", code);
    }

    #[test]
    fn mut_array_param_read_only_body_is_still_lent() {
        // The signature never depends on what the body does with the parameter.
        let src = "def int first(mut [int] a):\n    a[0]\n\ndef main():\n    mut [int] n = [1, 2]\n    print first(n)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn first(a: &mut Vec<isize>)"), "got:\n{}", code);
        assert!(code.contains("first(&mut n)"), "got:\n{}", code);
    }

    #[test]
    fn mut_array_param_on_struct_method_is_lent_not_moved() {
        let src = "struct Src:\n    var int pos = 0\n\n    def fill(mut [uint8] b):\n        b[0] = 9 as uint8\n\ndef main():\n    mut Src s = Src()\n    mut [uint8] buf = [0 as uint8 for i in 0..<4]\n    s.fill(buf)\n    print buf[0]\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn fill(&mut self, b: &mut Vec<u8>)"), "got:\n{}", code);
        assert!(code.contains("s.fill(&mut buf);"), "call site must lend `&mut buf` (not move `buf`), got:\n{}", code);
    }

    #[test]
    fn mut_dict_and_set_params_are_lent() {
        let src = "def tag(mut {string=int} d):\n    d[\"k\"] = 1\n\ndef int size(mut {int} s):\n    s.length\n\ndef main():\n    mut {string=int} d = {\"a\" = 0}\n    mut {int} s = {1}\n    tag(d)\n    print size(s)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn tag(d: &mut HashMap<Arc<str>, isize>)"), "got:\n{}", code);
        assert!(code.contains("tag(&mut d);"), "got:\n{}", code);
        assert!(code.contains("fn size(s: &mut HashSet<isize>)"), "got:\n{}", code);
        assert!(code.contains("size(&mut s)"), "got:\n{}", code);
    }

    #[test]
    fn mut_array_param_forwarded_to_method_is_reborrowed() {
        let src = "struct Src:\n    var int pos = 0\n\n    def fill(mut [uint8] b):\n        b[0] = 9 as uint8\n\ndef drive(mut Src s, mut [uint8] b):\n    s.fill(b)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("s.fill(&mut *b)"), "a lent param is reborrowed, not re-borrowed by `&mut b`, got:\n{}", code);
    }

    #[test]
    fn let_array_passed_to_mut_param_is_rejected_for_function_and_method() {
        let src = "def fill(mut [uint8] b):\n    b[0] = 7 as uint8\n\nstruct Src:\n    var int pos = 0\n\n    def fill_m(mut [uint8] b):\n        b[0] = 9 as uint8\n\ndef main():\n    mut Src s = Src()\n    let [uint8] buf = [0 as uint8 for i in 0..<4]\n    fill(buf)\n    s.fill_m(buf)\n";
        let errs = transpile_src_errors(src);
        assert_eq!(errs.matches("cannot pass `buf` to a `mut` parameter").count(), 2,
            "both the function and the method call must reject the `let` binding, got: {}", errs);
    }

    #[test]
    fn var_mut_param_form_parses_and_stays_a_rebindable_out_param() {
        let src = "def f(var mut [int] a):\n    a[0] = 1\n\ndef main():\n    var [int] n = [0]\n    f(n)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn f(") && code.contains("&mut Vec<isize>"), "got:\n{}", code);
        assert!(code.contains("f(&mut n)"), "got:\n{}", code);
    }

    #[test]
    fn external_non_buffer_method_still_takes_var_array_by_value() {
        // Only the listed buffer-filling names borrow: `tx.send(buf)` genuinely moves the Vec.
        let src = "use std.sync.mpsc\n\ndef f(mut [uint8] b, mut Sender tx) throws:\n    let r = try? tx.send(b)\n    guard let r else throw \"x\"\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(!code.contains("tx.send(&mut"),
            "a non-buffer method must keep by-value emission, got:\n{}", code);
    }

    #[test]
    fn external_static_call_borrows_non_literal_string_but_not_literal() {
        let src = "use std.net.TcpListener\nuse std.path.PathBuf\n\ndef main() throws:\n    let port = 0\n    let addr = \"127.0.0.1:{port}\"\n    let a = try? TcpListener.bind(addr)\n    let b = try? TcpListener.bind(\"127.0.0.1:0\")\n    let path = \"tmp/{port}\"\n    let text = try? std.fs.readToString(path)\n    var PathBuf base = PathBuf.new()\n    base.push(path)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("TcpListener::bind((&*addr))"), "got:\n{}", code);
        assert!(code.contains("TcpListener::bind(\"127.0.0.1:0\")"), "got:\n{}", code);
        assert!(code.contains("std::fs::read_to_string((&*path))"), "got:\n{}", code);
        assert!(code.contains("base.push((&*path))"), "got:\n{}", code);
    }

    #[test]
    fn fs_path_string_args_are_borrowed_in_every_shape() {
        // E0277 (`Arc<str>: AsRef<Path>`): a `string` local bound to a call/concatenation, and a
        // `string` passed to an imported free function, used to be passed by value.
        let src = "use std.fs.File\nuse std.fs.create_dir_all\nuse std.fs.remove_file\n\nstring name_of(int i):\n    \"/tmp/rep_{i}.bin\"\n\ndef f(string dir) throws:\n    let path = name_of(1)\n    let a = try? File.create(path)\n    let joined = dir + \"/x\"\n    let b = try? File.open(joined)\n    let c = try? create_dir_all(dir)\n    let d = try? remove_file(name_of(2))\n    let e = try? remove_file(path)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("File::create((&*path))"), "got:\n{}", code);
        assert!(code.contains("File::open((&*joined))"), "got:\n{}", code);
        assert!(code.contains("create_dir_all((&*dir))"), "got:\n{}", code);
        assert!(code.contains("remove_file((&*name_of(2)))"), "got:\n{}", code);
        assert!(code.contains("remove_file((&*path))"), "got:\n{}", code);
    }

    #[test]
    fn string_local_tracking_does_not_leak_into_a_later_same_named_int() {
        // `string_vars` is a flat per-fn, name-keyed set: `let si = to_s(x)` (string) must not
        // make a later `var si = 1` in the same fn a "string" (`si = si + 1` as concatenation).
        let src = "string to_s(int x):\n    \"{x}\"\n\ndef f(int n):\n    if n > 0:\n        let si = to_s(n)\n        print si\n    else:\n        var si = 1\n        while si < n:\n            si = si + 1\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(!code.contains("(&*si)") && !code.contains("si = Rc::<str>::from") && !code.contains("si = Arc::<str>::from"),
            "got:\n{}", code);
        assert!(code.contains("si += 1") || code.contains("si = (si + 1)") || code.contains("si = si + 1"), "got:\n{}", code);
    }

    #[test]
    fn explicit_external_metadata_can_keep_owned_string_argument() {
        let src = "use std.net.TcpListener\n\ndef main() throws:\n    let port = 0\n    let addr = \"127.0.0.1:{port}\"\n    let a = try? TcpListener.bind(addr)\n";
        let mut config = TranspileConfig::default();
        config.external_fns.push(("TcpListener".into(), "bind".into(), vec!["".into()]));
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("TcpListener::bind(addr)"), "got:\n{}", code);
        assert!(!code.contains("TcpListener::bind((&*addr))"), "got:\n{}", code);
    }

    #[test]
    fn optional_non_partial_eq_value_compared_with_nil_uses_presence_predicate() {
        let src = "use std.net.TcpListener\n\ndef main() throws:\n    let l = try? TcpListener.bind(\"127.0.0.1:0\")\n    print l != nil\n    print nil == l\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("l.is_some()"), "got:\n{}", code);
        assert!(code.contains("l.is_none()"), "got:\n{}", code);
        assert!(!code.contains("l != None") && !code.contains("None == l"), "got:\n{}", code);
    }

    #[test]
    fn external_read_user_struct_method_is_untouched() {
        // A Boring-declared `read` is not an external call: the *external-buffer* rule must not
        // fire — it is the method's own `var` param that lends `&mut buf` (see
        // `user_struct_method_var_array_param_is_lent_mutably`), not a name-keyed guess.
        let src = "struct Src:\n    var int pos = 0\n\n    def int read([uint8] b):\n        pos += 1\n        pos\n\ndef main():\n    var mut Src s = Src()\n    var [uint8] buf = [0 as uint8 for i in 0..<4]\n    let n = s.read(buf)\n    print n\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("s.read(") && !code.contains("s.read(&mut buf)"),
            "user-declared `read` with a by-value param must not be rewritten to `&mut`, got:\n{}", code);
    }

    // ── `var` out-parameter on a user-struct *method* ────────────────────────────────────
    // Free functions already lend `&mut <place>` for a `var` param (`emit_args_coerced`); the
    // user-struct-receiver arm of `emit_method_call_fallback` used to emit the by-value clone
    // (`s.read(buf.clone())`) against a `&mut Vec<u8>` signature. See
    // `tests/cases/method_var_param.br` for the same through a real `cargo run`.

    #[test]
    fn user_struct_method_var_array_param_is_lent_mutably() {
        let src = "struct Src:\n    var int pos = 0\n\n    def int read(var [uint8] b):\n        pos += 1\n        pos\n\ndef main():\n    var mut Src s = Src()\n    var [uint8] buf = [0 as uint8 for i in 0..<4]\n    let n = s.read(buf)\n    print n\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("s.read(&mut buf)"),
            "`var [uint8]` method param must be passed as `&mut buf`, got:\n{}", code);
        assert!(!code.contains("buf.clone()"),
            "the by-value clone must be stripped for an out-parameter, got:\n{}", code);
    }

    #[test]
    fn user_struct_method_var_param_forwarded_from_var_param_is_reborrowed() {
        let src = "struct Src:\n    var int pos = 0\n\n    def int read(var [uint8] b):\n        pos += 1\n        pos\n\nint pump(mut Src s, var [uint8] b):\n    s.read(b)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("s.read(&mut ") && !code.contains("s.read(b.clone())") && !code.contains("(*b).clone()"),
            "a forwarded `var` param must be reborrowed, not cloned, got:\n{}", code);
    }

    #[test]
    fn user_struct_method_var_param_still_rejects_let_argument() {
        let src = "struct Src:\n    var int pos = 0\n\n    def int read(var [uint8] b):\n        pos += 1\n        pos\n\ndef main():\n    var mut Src s = Src()\n    let [uint8] buf = [0 as uint8 for i in 0..<4]\n    let n = s.read(buf)\n    print n\n";
        let err = transpile_src_errors(src);
        assert!(err.contains("cannot pass `buf` to a `var` out-parameter"),
            "a `let` argument to a `var` method param must keep the out-parameter diagnostic, got:\n{}", err);
    }

    // ── user-struct method named like a builtin `USIZE_INDEX_METHODS` entry ─────────────────
    // `emit_method_call_fallback` used to cast the first argument `as usize` for any method
    // named `nth`/`insert`/`swap`/... whenever the receiver wasn't a set/dict, even a
    // Boring-declared struct's own method (`s.swap((&mut buf as usize))` -- invalid cast).

    #[test]
    fn user_struct_method_named_swap_with_var_array_param_has_no_usize_cast() {
        let src = "struct Src:\n    var int pos = 0\n\n    def int swap(var [uint8] b):\n        pos += 1\n        b = [1 as uint8, 2 as uint8]\n        pos\n\ndef main():\n    var mut Src s = Src()\n    var [uint8] buf = [0 as uint8 for i in 0..<4]\n    let m = s.swap(buf)\n    print m\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("s.swap(&mut buf)") && !code.contains("buf as usize"),
            "user-declared `swap` must keep its `var` out-param borrow with no usize cast, got:\n{}", code);
    }

    #[test]
    fn user_struct_method_named_insert_keeps_int_arg_as_isize() {
        let src = "struct Log:\n    var int last = 0\n\n    def insert(int idx, string s):\n        last = idx\n\ndef main():\n    var mut Log l = Log()\n    let int i = 3\n    l.insert(i, \"x\")\n    l.insert(-1, \"y\")\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(!code.contains("i as usize") && !code.contains("(-1) as usize") && !code.contains("-1 as usize"),
            "user-declared `insert(int idx, ...)` must not get its `int` arg cast to usize, got:\n{}", code);
    }

    #[test]
    fn builtin_vec_swap_and_insert_still_cast_index_to_usize() {
        let src = "def main():\n    var [int] v = [1, 2, 3]\n    let int i = 0\n    v.swap(i, 1)\n    v.insert(i, 9)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("v.swap((i as usize)") && code.contains("v.insert((i as usize)"),
            "builtin Vec swap/insert must keep the usize index cast, got:\n{}", code);
    }

    // ── builtin trailing `as <type>` as the left operand of `<` / `<<` ───────────────────
    // `ord(c) < 32` was emitted `(c).chars().next().expect(..) as isize < 32`, which rustc
    // parses as generic arguments for `isize`.

    #[test]
    fn ord_is_parenthesised_as_left_operand_of_lt_and_shl() {
        let src = "def bool is_ctrl(string ch):\n    ord(ch) < 32\n\ndef int shl(string ch):\n    ord(ch) << 1\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(!code.contains("as isize <"),
            "`ord(..) as isize` must be parenthesised before `<`/`<<`, got:\n{}", code);
        assert!(code.contains("as isize) < 32") && code.contains("as isize) << 1"),
            "expected parenthesised ord cast, got:\n{}", code);
    }

    #[test]
    fn plain_cast_and_count_closure_are_parenthesised_as_left_operand_of_lt() {
        let src = "def main():\n    let n = 5\n    let [int] xs = [1, 2]\n    let a = (n as int) < 9\n    let b = xs.count((x): x > 1) < 5\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(!code.contains("as isize <") && !code.contains("as i64 <"),
            "casts must be parenthesised before `<`, got:\n{}", code);
    }

    #[test]
    fn non_clone_std_type_arg_is_moved_not_cloned() {
        let src = "use std.net.TcpStream\n\ndef int handle(TcpStream s) throws:\n    return 1\n\ndef int serve(TcpStream sock) throws:\n    return handle(sock)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("handle(sock)?") && !code.contains("sock.clone()"),
            "TcpStream is not Clone — must be moved into the call, got:\n{}", code);
    }

    #[test]
    fn non_clone_std_type_field_read_let_is_moved_not_cloned() {
        // `var TcpStream server = pair.0` — the declared type is a non-`Clone` std handle, so the
        // let-binding's auto field-read `.clone()` (E0599) must be suppressed.
        let src = "use std.net.TcpListener, TcpStream\n\ndef int f(TcpListener l) throws:\n    let p = try? l.accept()\n    guard let pair = p else throw \"x\"\n    var TcpStream server = pair.0\n    return 1\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("let mut server: TcpStream = pair.0;"),
            "TcpStream field read must be a move, got:\n{}", code);
        assert!(!code.contains("pair.0.clone()"), "got:\n{}", code);
    }

    #[test]
    fn non_clone_std_type_untyped_tuple_field_read_is_moved_not_cloned() {
        // No declared type: `pair`'s tuple type comes from the `TcpListener.bind` -> `accept()`
        // table (`infer_std_type`), through `try?` and `guard let`.
        let src = "use std.net.TcpListener, TcpStream\n\ndef int f() throws:\n    let l = try? TcpListener.bind(\"127.0.0.1:0\")\n    guard let listener = l else throw \"x\"\n    let p = try? listener.accept()\n    guard let pair = p else throw \"y\"\n    let server = pair.0\n    let peer = pair.1\n    return 1\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("let server = pair.0;"), "TcpStream element must be moved, got:\n{}", code);
        assert!(code.contains("let peer = pair.1.clone();"),
            "the SocketAddr element is Clone and keeps the field-read clone, got:\n{}", code);
    }

    #[test]
    fn non_clone_std_type_untyped_tuple_field_read_from_declared_listener_param() {
        let src = "use std.net.TcpListener\n\ndef int f(TcpListener l) throws:\n    let p = try? l.accept()\n    guard let pair = p else throw \"y\"\n    var server = pair.0\n    return 1\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("let mut server = pair.0;") && !code.contains("pair.0.clone()"), "got:\n{}", code);
    }

    #[test]
    fn non_clone_std_type_untyped_child_stdio_field_read_is_moved() {
        let src = "use std.process.Command\n\ndef int f() throws:\n    let c = try? Command.new(\"cat\").spawn()\n    guard let child = c else throw \"x\"\n    var pipe = child.stdin\n    return 1\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("let mut pipe = child.stdin;") && !code.contains("child.stdin.clone()"), "got:\n{}", code);
    }

    #[test]
    fn user_type_named_like_std_listener_untyped_tuple_field_read_still_clones() {
        // A Boring `struct TcpListener` shadows std's: `accept()` is the user's own method, so the
        // std return table must not apply and the value-semantics clone stays.
        let src = "struct Conn:\n    int fd\n\nstruct TcpListener:\n    int x\n\n    def (Conn, int) accept():\n        (Conn(1), 2)\n\ndef f(TcpListener l):\n    let pair = l.accept()\n    let c = pair.0\n    print c.fd\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(!code.contains("let c = pair.0;"), "user type must keep the clone, got:\n{}", code);
    }

    #[test]
    fn untyped_tuple_field_read_of_unknown_call_still_clones() {
        // Control: an untyped tuple from a call outside the std table keeps the auto clone.
        let src = "def (string, int) mk():\n    (\"a\", 1)\n\ndef f():\n    let pair = mk()\n    let s = pair.0\n    print s\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("pair.0.clone()"), "got:\n{}", code);
    }

    #[test]
    fn clone_type_field_read_let_still_clones() {
        // Control: `Duration` / user structs are `Clone` and keep the value-semantics clone.
        let src = "use std.time.Duration\n\nstruct P:\n    int x\n\nstruct H:\n    P p\n    Duration d\n\ndef f(H h):\n    let P q = h.p\n    let Duration e = h.d\n    print q.x\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("h.p.clone()") && code.contains("h.d.clone()"),
            "Clone types must keep the field-read clone, got:\n{}", code);
    }

    #[test]
    fn user_struct_shadowing_non_clone_name_still_clones_on_field_read() {
        // A Boring `struct File` is a user type, not std's `File`: the exemption must not apply.
        let src = "struct File:\n    int fd\n\nstruct H:\n    File f\n\ndef g(H h):\n    let File x = h.f\n    print x.fd\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("h.f.clone()"), "got:\n{}", code);
    }

    // ── docs/option-return-double-some-wrap-bug.md ────────────────────────────
    // A `T?`-returning function/method (or a plain assignment into an already-
    // `T?`-typed local) must NOT wrap an expression that is itself statically
    // Option-shaped (a `T?`-declared field read, or a call to another
    // `T?`-returning function/method) in an extra `Some(...)` — that produces
    // `Option<Option<T>>`, a real E0308. These four tests cover the doc's three
    // repros (tail-position field read, tail-position method call, plain
    // assignment) plus the still-needed-wrap case that must keep working.

    #[test]
    fn optional_return_of_optional_field_is_not_double_wrapped() {
        // Repro 1: `req string? next_id(): next` where `next` is a `string?` field.
        let src = "\
struct Block:\n    string? next\n\n    req string? next_id():\n        next\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn next_id(&self) -> Option<Arc<str>> {\n        self.next.clone()\n    }"),
            "tail-position read of a T?-declared field must be passed through raw, not Some(...)-wrapped, got:\n{}", code);
        assert!(!code.contains("Some(self.next"),
            "must not double-wrap an already-Option field read, got:\n{}", code);
    }

    #[test]
    fn optional_return_of_optional_method_call_is_not_double_wrapped() {
        // Repro 2: `def string? first_of([JVal] items): items[0].as_str()` where
        // `as_str()` is itself declared `string?` — receiver is an `Index` expr,
        // not a bare `Var`, so this also exercises the general (non-Var-only)
        // receiver resolution.
        let src = "\
enum JVal:\n    JStr(string)\n    JNum(float)\n\n    req string? as_str():\n        match self.clone():\n            JStr(s): s\n            _: nil\n\n\
def string? first_of([JVal] items):\n    items[0].as_str()\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn first_of(items: &Vec<JVal>) -> Option<Arc<str>> {\n    items[(0) as usize].as_str()\n}"),
            "tail call to a T?-returning method must be passed through raw, not Some(...)-wrapped, got:\n{}", code);
        assert!(!code.contains("Some(items"),
            "must not double-wrap an already-Option method-call result, got:\n{}", code);
    }

    #[test]
    fn optional_assignment_of_optional_method_call_is_not_double_wrapped() {
        // Repro 3: `id = items[0].as_str()` assigned into a `var string? id`.
        let src = "\
enum JVal:\n    JStr(string)\n    JNum(float)\n\n    req string? as_str():\n        match self.clone():\n            JStr(s): s\n            _: nil\n\n\
def string? parse([JVal] items):\n    var string? id = nil\n    id = items[0].as_str()\n    id\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("id = items[(0) as usize].as_str();"),
            "assignment RHS that's already T?-shaped must not be Some(...)-wrapped, got:\n{}", code);
        assert!(!code.contains("Some(items"),
            "must not double-wrap an already-Option assignment RHS, got:\n{}", code);
    }

    #[test]
    fn optional_return_of_bare_value_is_still_wrapped() {
        // Non-regression: a genuinely bare `T` (a plain, non-optional field) in
        // T?-returning tail position must still get Some(...) — this is the
        // legitimate case the fix must not break.
        let src = "\
struct Block:\n    string label\n\n    req string? label_opt():\n        label\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("Some(self.label"),
            "a bare (non-optional) field read must still be wrapped in Some(...), got:\n{}", code);
    }

    #[test]
    fn plain_value_assign_to_external_type_optional_field_is_wrapped() {
        // Follow-up to e0247ff: `target_field_ty` (`emit_expr_assign`) only ever consulted
        // `struct_fields`, which exists solely for a Boring-*declared* struct — a field on
        // an EXTERNAL type (Boring never parsed a `struct` for it) still missed the
        // `Some(...)` wrap. `Foo` here is never declared as a Boring struct at all (there's
        // no `struct Foo:` anywhere in `src`), standing in for a real external type like
        // Bevy's `Sprite`; `custom_thing` is registered only via the `boring.toml
        // [external_types] optional_fields` config-supplement path (`external_optional_fields`),
        // not the built-in `KNOWN_EXTERNAL_OPTIONAL_FIELDS` table — the real Bevy
        // `Sprite`/`custom_size` case is covered end-to-end (a real `cargo build`) by the
        // `ext_optional_field_assign` project test in tests/transpile.rs instead.
        let src = "\
def sync():\n    var mut Foo f = Foo(x = 1, _)\n    f.custom_thing = 5\n";
        let config = TranspileConfig {
            external_optional_fields: vec![("Foo".to_string(), "custom_thing".to_string())],
            ..TranspileConfig::default()
        };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("f.custom_thing = Some(5);"),
            "a plain-value assignment to a whitelisted external type's Option<T> field must be \
             Some(...)-wrapped, got:\n{}", code);
    }

    #[test]
    fn introspect_struct_impl_synthesizes_typed_field_lookup() {
        // v2 (handle-based) API — see docs/book.md "Introspect — read-only reflection"
        // and the `boring_introspect_trait_design` memo's "FINAL v2 design".
        let src = "\
struct Point as Introspect:\n    float x\n    string label\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("trait Introspect: std::any::Any {"),
            "the Introspect trait prelude must be emitted once uses_introspect is set, got:\n{}", code);
        assert!(code.contains("impl Introspect for Point {"),
            "struct header `as Introspect` must synthesize a manual impl (never a #[derive(...)]), got:\n{}", code);
        assert!(code.contains("fn introspect(&self) -> &'static IntrospectInfo {"), "got:\n{}", code);
        assert!(code.contains("static INFO: std::sync::OnceLock<IntrospectInfo> = std::sync::OnceLock::new();")
            && code.contains("INFO.get_or_init(|| IntrospectInfo {"),
            "introspect() must be backed by a per-type OnceLock, computed once and returned by \
             reference, not a fresh Vec<Field>/Vec<Method> allocation on every call, got:\n{}", code);
        assert!(code.contains("typeName: Arc::from(\"Point\"),") || code.contains("typeName: Rc::from(\"Point\"),"),
            "got:\n{}", code);
        assert!(code.contains("Ok(FieldValue::Float(s.x as f64))"),
            "a lowercase `float` field must resolve via normalize_type_name to the typed FieldValue::Float variant, not the Other/Debug fallback, got:\n{}", code);
        assert!(code.contains("Ok(FieldValue::Str(s.label.clone()))"), "got:\n{}", code);
        assert!(code.contains("fn get(&self, instance: &dyn Introspect) -> Result<FieldValue, IntrospectError> {"),
            "Field::get(instance) is the instance-scoped form of the two-arity get() overload, got:\n{}", code);
        assert!(code.contains("variantIndex: 0,"),
            "a struct is its own single permanent \"variant\" for the uniform dyn Introspect interface, got:\n{}", code);
        assert!(!code.lines().any(|l| l.contains("#[derive(") && l.contains("Introspect")),
            "Introspect must never be folded into a #[derive(...)] list (no proc macro backs it), got:\n{}", code);
    }

    #[test]
    fn introspect_nested_field_qualifies_when_target_implements_introspect_and_clone() {
        // `Nested`/`Actor`/`Guard` FieldValue variants (see the `boring_introspect_trait_design`
        // memo's "Nested/Actor/Guard FieldValue variants" section) — an `'inline` (bare, no
        // qualifier) field whose own type declares `as Introspect` AND derives `Clone`
        // (the default for a plain struct with no non-Clone field) gets `FieldValue::Nested`,
        // not the `Other`/Debug fallback.
        let src = "\
struct Inner as Introspect:\n    int v\n\n\
struct Outer as Introspect:\n    Inner nested\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("Ok(FieldValue::Nested(Box::new(s.nested.clone())))"),
            "a bare (implicit 'inline) Introspect+Clone-implementing field type must produce \
             FieldValue::Nested, got:\n{}", code);
    }

    #[test]
    fn introspect_nested_field_falls_back_to_other_when_target_not_clone() {
        // Same shape as above, but `Inner` has an explicit `@derive(...)` that omits `Clone`
        // — `Box::new(access.clone())` wouldn't compile (`Inner: Clone` doesn't hold), so this
        // must keep falling back to `Other`/Debug exactly as before this feature, not emit code
        // that would fail a real `cargo build`.
        let src = "\
@derive(Debug)\n\
struct Inner as Introspect:\n    int v\n\n\
struct Outer as Introspect:\n    Inner nested\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("Ok(FieldValue::Other(format!(\"{:?}\", s.nested)))"),
            "an Introspect-implementing but non-Clone target type must fall back to Other, got:\n{}", code);
        assert!(!code.contains("Ok(FieldValue::Nested("), "got:\n{}", code);
    }

    #[test]
    fn introspect_nested_field_falls_back_to_other_when_target_not_introspect() {
        // `Inner` doesn't declare `as Introspect` at all — purely additive, zero regression:
        // must behave exactly like every non-qualifying field did before this feature existed.
        let src = "\
struct Inner:\n    int v\n\n\
struct Outer as Introspect:\n    Inner nested\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("Ok(FieldValue::Other(format!(\"{:?}\", s.nested)))"),
            "a field whose type doesn't declare as Introspect at all must fall back to Other, got:\n{}", code);
    }

    #[test]
    fn introspect_owned_shared_actor_guard_fields_produce_the_right_variant_and_shape() {
        // One field per qualifier group — see `introspect_nested_field_value_expr`'s doc
        // comment for the exact Rust shape each one must produce.
        let src = "\
struct Inner as Introspect:\n    int v\n\n\
struct Outer as Introspect:\n    \
Inner'owned owned_f\n    Inner'shared shared_f\n    Inner'actor actor_f\n    Inner'guard guard_f\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        // 'owned (Box<Inner>) — access.clone() is already Box<Inner>, no rebox.
        assert!(code.contains("Ok(FieldValue::Nested(s.owned_f.clone()))"),
            "'owned must coerce its existing Box<T> directly, no Box::new(...) rebox, got:\n{}", code);
        // 'shared (Rc<Inner>/Arc<Inner>) — Box::new(cheap handle clone), no T: Clone needed.
        assert!(code.contains("Ok(FieldValue::Nested(Box::new(s.shared_f.clone())))"), "got:\n{}", code);
        // 'actor → FieldValue::Actor, same Box::new(handle clone) shape.
        assert!(code.contains("Ok(FieldValue::Actor(Box::new(s.actor_f.clone())))"), "got:\n{}", code);
        // 'guard → FieldValue::Guard.
        assert!(code.contains("Ok(FieldValue::Guard(Box::new(s.guard_f.clone())))"), "got:\n{}", code);
    }

    #[test]
    fn introspect_struct_fields_only_enum_variant_fields_keep_falling_back_to_other() {
        // Explicit scope cut (design memo): enum variant fields are NOT extended this round,
        // even when the field's own type declares `as Introspect` and is `Clone` — a struct
        // field with the identical type would get `FieldValue::Nested`.
        let src = "\
struct Inner as Introspect:\n    int v\n\n\
enum Shape as Introspect:\n    Circle(Inner center)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("FieldValue::Other(format!(\"{:?}\", "),
            "enum variant fields must keep falling back to Other this round, got:\n{}", code);
        assert!(!code.contains("Ok(FieldValue::Nested("), "got:\n{}", code);
    }

    #[test]
    fn introspect_type_level_field_gets_nested_treatment_for_free() {
        // "Type-level `type let` fields going through the same struct-field codegen path get
        // this for free" — per the design memo's scope note.
        let src = "\
struct Inner as Introspect:\n    int v\n\n\
struct Outer as Introspect:\n    type let Inner origin = Inner(v = 0)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("Ok(FieldValue::Nested(Box::new(Outer::ORIGIN.clone())))"),
            "a type-level `type let` field of a Clone+Introspect type must also get Nested, got:\n{}", code);
    }

    #[test]
    fn introspect_rc_and_box_wrapper_delegates_introspect_as_any_through_deref() {
        // The blanket `Rc<T>`/`Box<T>` (single-threaded default in tests) delegate impls must
        // forward `__introspectAsAny`/`Mut` through `(**self)` — NOT return `self` — or a
        // recursive `Field.get()` on a `'shared`/`'owned` nested value would always downcast
        // against the wrapper type instead of the real `T` and fail every time.
        let src = "struct Point as Introspect:\n    int v\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("impl<T: Introspect + ?Sized> Introspect for Rc<T> {")
            || code.contains("impl<T: Introspect + ?Sized> Introspect for Arc<T> {"),
            "got:\n{}", code);
        assert!(code.contains("impl<T: Introspect + ?Sized> Introspect for Box<T> {"), "got:\n{}", code);
        // Every strong (Rc/Arc/Box) impl's __introspectAsAny must delegate through (**self).
        let strong_as_any_lines: Vec<&str> = code.lines()
            .filter(|l| l.contains("fn __introspectAsAny(&self)"))
            .filter(|l| l.contains("(**self)"))
            .collect();
        assert!(!strong_as_any_lines.is_empty(),
            "at least the Rc/Arc/Box blanket impls must delegate __introspectAsAny through (**self), got:\n{}", code);
    }

    #[test]
    fn weak_annotated_local_upgrade_is_unwrapped() {
        // Regression: a local declared `T'shared'weak` was resolved to the user struct `T`
        // by name, so `.upgrade()` was dispatched verbatim as a user-struct method and kept
        // its `Option` (`upgraded.val` then failed with E0609 on `Option<Arc<T>>`).
        let src = "struct WNode:\n    init(pub int val)\n\nlet WNode'shared strong = WNode(val = 42)\nlet WNode'shared'weak weakn = strong\nlet upgraded = weakn.upgrade()\nprint upgraded.val\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("weakn.upgrade().expect("), "got:\n{}", code);
    }

    #[test]
    fn introspect_actor_guard_weak_wrappers_do_not_delegate_as_any() {
        // `RefCell`/`Mutex`/`RwLock` (single-threaded test → RefCell) deliberately do NOT
        // delegate `__introspectAsAny`/`Mut` — they return the wrapper itself, `self`,
        // untouched (the deliberate, documented degraded behavior for 'actor/'guard: shape
        // visible via introspect(), but a Field/Method resolved from that shape and reused
        // against the same value fails the ordinary "wrong instance type" check, not a panic).
        let src = "struct Point as Introspect:\n    int v\n";
        let code = transpile_src_with_config(
            src,
            TranspileConfig { threading: crate::transpiler::ThreadingMode::Single, ..TranspileConfig::default() },
        );
        assert!(code.contains("impl<T: Introspect> Introspect for RefCell<T> {"), "got:\n{}", code);
        let refcell_block_start = code.find("impl<T: Introspect> Introspect for RefCell<T> {")
            .unwrap_or_else(|| panic!("no RefCell impl found, got:\n{}", code));
        let refcell_block = &code[refcell_block_start..];
        let block_end = refcell_block.find("\n}\n").map(|i| i + 3).unwrap_or(refcell_block.len());
        let refcell_block = &refcell_block[..block_end];
        assert!(refcell_block.contains("fn __introspectAsAny(&self) -> &dyn std::any::Any { self }"),
            "RefCell<T>'s __introspectAsAny must return `self` unchanged, not delegate, got:\n{}", refcell_block);
        assert!(refcell_block.contains("fn __introspectAsAnyMut(&mut self) -> &mut dyn std::any::Any { self }"),
            "got:\n{}", refcell_block);
        assert!(refcell_block.contains("self.try_borrow()"),
            "introspect() itself DOES delegate (borrow, call, release — the &'static result \
             outlives the guard trivially), got:\n{}", refcell_block);
    }

    #[test]
    fn introspect_rebindable_flag_matches_actual_setter_availability() {
        // Regression for a real bug found post-ship: a `var`-qualified field whose type
        // doesn't round-trip through `FieldValue` (anything falling back to `Other`/Debug —
        // a nested struct here) got no setter fn generated, but `isRebindable` still echoed
        // the raw language-level `var`-ness (`true`), so `Field.set()` would throw "has no
        // setter" despite `isRebindable` promising it would work. `isRebindable` must be
        // gated on setter-generation succeeding, not just on the field's own `var`/`mut`.
        let src = "\
struct Inner:\n    float v\n\n\
struct Outer as Introspect:\n    var Inner nested\n    var int count\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("name: Arc::from(\"nested\")") || code.contains("name: Rc::from(\"nested\")"),
            "got:\n{}", code);
        let nested_line = code.lines().find(|l| l.contains("name: Arc::from(\"nested\")") || l.contains("name: Rc::from(\"nested\")"))
            .unwrap_or_else(|| panic!("no Field entry for 'nested', got:\n{}", code));
        assert!(nested_line.contains("isRebindable: false"),
            "a `var`-qualified field whose type falls back to Other/Debug must report isRebindable: false (no setter can be generated for it), got line:\n{}", nested_line);
        assert!(nested_line.contains("setter: None"), "got line:\n{}", nested_line);
        let count_line = code.lines().find(|l| l.contains("name: Arc::from(\"count\")") || l.contains("name: Rc::from(\"count\")"))
            .unwrap_or_else(|| panic!("no Field entry for 'count', got:\n{}", code));
        assert!(count_line.contains("isRebindable: true"),
            "a `var`-qualified primitive field must still report isRebindable: true (the fix must not regress the working case), got line:\n{}", count_line);
        assert!(count_line.contains("setter: Some("), "got line:\n{}", count_line);
    }

    #[test]
    fn introspect_struct_impl_covers_init_only_fields() {
        // Fields declared only via a no-body `init` (no `struct X: <fields>` block) must
        // still be picked up — see `emit_introspect_struct_impl`'s doc comment.
        let src = "struct Vec2 as Introspect:\n    init(pub float x, pub float y)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("name: Arc::from(\"x\")") || code.contains("name: Rc::from(\"x\")"),
            "init-only struct fields must surface in introspect()'s fields list, got:\n{}", code);
        assert!(code.contains("Ok(FieldValue::Float(s.x as f64))"), "got:\n{}", code);
    }

    #[test]
    fn introspect_enum_impl_matches_per_variant() {
        let src = "\
enum Shape as Introspect:\n    Circle(float radius)\n    Point\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("impl Introspect for Shape {"), "got:\n{}", code);
        assert!(code.contains("fn introspect(&self) -> &'static IntrospectInfo {")
            && code.contains("static VARIANTS: std::sync::OnceLock<Vec<IntrospectInfo>> = std::sync::OnceLock::new();"),
            "introspect() is backed by a per-type OnceLock<Vec<IntrospectInfo>> (one entry per \
             variant), not rebuilt per call, got:\n{}", code);
        assert!(code.contains("variantName: Arc::from(\"Circle\"), variantIndex: 0, fields: vec![")
            || code.contains("variantName: Rc::from(\"Circle\"), variantIndex: 0, fields: vec!["),
            "introspect() must precompute one IntrospectInfo entry per variant, got:\n{}", code);
        assert!(code.contains("variantName: Arc::from(\"Point\"), variantIndex: 1, fields: vec![], methods:")
            || code.contains("variantName: Rc::from(\"Point\"), variantIndex: 1, fields: vec![], methods:"),
            "a unit variant's entry must have an empty fields list, got:\n{}", code);
        assert!(code.contains("Shape::Circle(f0) => 0,") && code.contains("Shape::Point => 1,"),
            "the runtime match only needs to resolve which precomputed entry to index into, got:\n{}", code);
        assert!(code.contains("Ok(FieldValue::Float(*f0 as f64))"),
            "an enum match-arm-bound field is `&T` under match ergonomics and needs a leading deref, got:\n{}", code);
        assert!(code.contains("fn __introspectGet_Shape_Circle_radius("),
            "each named variant field gets its own generated getter free fn, got:\n{}", code);
    }

    #[test]
    fn introspect_handle_call_does_not_double_reference_an_already_borrowed_param() {
        // Regression: a struct/enum-typed function PARAMETER with no explicit qualifier is
        // ALREADY `&T` in the generated Rust (Boring's default auto-ref parameter-passing
        // convention, see `emit_param`) — `f.get(&(s))` where `s` is such a parameter
        // produces `&&Shape`, which does NOT coerce to `&dyn Introspect` (confirmed via a
        // real `cargo build`, E0277 "the trait bound `&Shape: Introspect` is not
        // satisfied"). This was previously masked for enum params specifically by a SEPARATE
        // bug (enum protocols weren't registered in `struct_protocols` at all, so a header-
        // only trait method like `introspect()` was always misjudged "mutating," forcing the
        // param to stay owned) — fixing that unmasked this one, found via a real
        // `examples/introspect_enum_demo.br` recompile.
        let src = "\
enum Shape as Introspect:\n    Circle(float radius)\n\n\
def describe(Shape s) throws:\n    let info = s.introspect()\n    for f in info.fields:\n        let v = f.get(s)\n        print \"{v:?}\"\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn describe(s: &Shape)"),
            "sanity check: an unqualified enum param must be &T by default, got:\n{}", code);
        assert!(code.contains("f.get(s)?") || code.contains("f.get(s)"),
            "an already-&T instance argument must NOT be wrapped in another &(...), got:\n{}", code);
        assert!(!code.contains("f.get(&(s))"), "got:\n{}", code);
    }

    #[test]
    fn introspect_via_derive_attribute_is_rejected_with_a_clear_error() {
        // The natural first mistake for anyone used to `@derive(Debug, Clone, ...)` is to
        // reach for `@derive(Introspect)` too — it must be rejected with a clear message
        // pointing at `as Introspect`, not silently emit a broken `#[derive(Introspect)]`
        // rustc would reject downstream with a confusing "cannot find derive macro" error.
        let src = "@derive(Introspect)\nstruct Point:\n    float x\n";
        let tokens = crate::lexer::lex(src).expect("lex error");
        let program = crate::parser::parse(tokens).expect("parse error");
        let out = transpile_with_config(&program, TranspileConfig::default());
        assert!(!out.errors.is_empty(), "must reject @derive(Introspect)");
        assert!(out.errors[0].message.contains("as Introspect"),
            "error must point at the correct fix, got: {:?}", out.errors[0].message);
        assert!(!out.code.contains("#[derive(Introspect)]"),
            "must never leak a broken #[derive(Introspect)] into the generated Rust, got:\n{}", out.code);
    }

    #[test]
    fn introspect_as_bare_trait_type_is_boxed_dyn_dispatch() {
        // `Introspect` is a built-in trait with no `trait Introspect: ...` declaration of
        // its own, so it never reaches `trait_method_names` just by being parsed the way a
        // real user-declared trait's methods do — every "traits as types" dynamic-dispatch
        // consumer (emit_let.rs's array/scalar boxing, emit_loop.rs's borrow-vs-`.cloned()`
        // choice, emit_top.rs's `Box<dyn Trait>` parameter emission, emit_methods.rs's
        // camelCase-preservation check) gates on `trait_method_names.contains_key(name)`, so
        // `Introspect` must be seeded into that table directly (mod.rs's `Transpiler::new`) —
        // this is the regression that would fire if that seeding were ever dropped.
        let src = "\
struct Point as Introspect:\n    float x\n    float y\n    string label\n\n\
enum Shape as Introspect:\n    Circle(float radius)\n    Point\n\n\
def main():\n    let [Introspect] items = [Point(x = 1.0, y = 2.0, label = \"a\"), Shape.Circle(3.0)]\n    for item in items:\n        print item.typeName()\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("let items: Vec<Box<dyn Introspect>>"),
            "a `[Introspect]` local must be `Vec<Box<dyn Introspect>>`, got:\n{}", code);
        assert!(code.contains("Box::new(Point"), "array element must be boxed, got:\n{}", code);
        assert!(code.contains("Box::new(Shape::Circle"), "array element must be boxed, got:\n{}", code);
        assert!(!code.contains(".iter().cloned()"),
            "must not use .iter().cloned() over a Vec<Box<dyn Introspect>> (not Clone), got:\n{}", code);
        assert!(code.contains("for item in &items"), "must borrow when iterating, got:\n{}", code);
        assert!(code.contains("item.typeName()"),
            "camelCase call through a trait-typed receiver must be preserved, not mangled to \
             item.type_name(), got:\n{}", code);
    }

    #[test]
    fn introspect_variadic_param_method_is_visible_but_not_callable() {
        // "Method visibility expansion" — `info.methods` is exhaustive except `native`;
        // a variadic-parameter method now APPEARS (unlike the old `introspect_method_is_
        // reflectable`, which excluded it outright) but with `isCallable: false` and no
        // real call body — `MethodAccess`'s dispatch slot is `None`, not a generated fn.
        let src = "\
struct Widget as Introspect:\n    var int count\n\n    req int sumAll(int... nums):\n        count\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("name: Arc::from(\"sumAll\")") || code.contains("name: Rc::from(\"sumAll\")"),
            "a variadic-param method must still appear in the methods list, got:\n{}", code);
        let entry_line = code.lines().find(|l| l.contains("name: Arc::from(\"sumAll\")") || l.contains("name: Rc::from(\"sumAll\")"))
            .unwrap_or_else(|| panic!("no Method entry for 'sumAll', got:\n{}", code));
        assert!(entry_line.contains("isCallable: false"),
            "a variadic param can't round-trip through [FieldValue], so isCallable must be false, got line:\n{}", entry_line);
        assert!(entry_line.contains("kind: MethodKind::Plain"), "got line:\n{}", entry_line);
        assert!(entry_line.contains("MethodAccess::InstanceReq(None)"),
            "the dispatch slot must be None (no call body generated) to match isCallable: false, got line:\n{}", entry_line);
        assert!(!code.contains("__introspectCall_Widget_sumAll"),
            "no free dispatch fn should be generated for an uncallable method, got:\n{}", code);
    }

    #[test]
    fn introspect_throws_method_is_callable_and_propagates_via_question_mark() {
        // Real new capability (not just visibility): a `throws` method's generated
        // `request`/`modify` call body must actually call the underlying method and
        // propagate any error via `?` — safe because a `throws` method's real Rust return
        // type, `Result<T, Box<dyn std::error::Error + Send + Sync>>`, is exactly
        // `IntrospectError`'s alias target, so `?` needs no conversion.
        let src = "\
struct Widget as Introspect:\n    var int count\n\n    req int risky(int x) throws:\n        guard x != 0 else throw Error.InvalidInput\n        100 / x\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn __introspectCall_Widget_risky("),
            "a throws method must get a real generated call body now, got:\n{}", code);
        let fn_start = code.find("fn __introspectCall_Widget_risky(").unwrap_or_else(|| panic!("got:\n{}", code));
        let fn_body = &code[fn_start..];
        assert!(fn_body.contains("s.risky(a0)?"),
            "the underlying throws call must be `?`-propagated, got:\n{}", fn_body);
        let entry_line = code.lines().find(|l| l.contains("name: Arc::from(\"risky\")") || l.contains("name: Rc::from(\"risky\")"))
            .unwrap_or_else(|| panic!("no Method entry for 'risky', got:\n{}", code));
        assert!(entry_line.contains("isCallable: true"),
            "a throws method (typed params, not task/stream) must be callable, got line:\n{}", entry_line);
        assert!(entry_line.contains("MethodAccess::InstanceReq(Some("), "got line:\n{}", entry_line);
    }

    #[test]
    fn introspect_task_method_is_visible_but_not_callable() {
        // Same "visible, not callable" treatment as a variadic param, but for the
        // task/stream exclusion specifically — `MethodKind::Task` is reported too.
        let src = "\
struct Widget as Introspect:\n    var int count\n\n    task req int fetchCount():\n        count\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        let entry_line = code.lines().find(|l| l.contains("name: Arc::from(\"fetchCount\")") || l.contains("name: Rc::from(\"fetchCount\")"))
            .unwrap_or_else(|| panic!("no Method entry for 'fetchCount', got:\n{}", code));
        assert!(entry_line.contains("isCallable: false"), "got line:\n{}", entry_line);
        assert!(entry_line.contains("kind: MethodKind::Task"), "got line:\n{}", entry_line);
        assert!(entry_line.contains("MethodAccess::InstanceReq(None)"), "got line:\n{}", entry_line);
    }

    #[test]
    fn introspect_method_kind_expr_reports_task_stream_plain() {
        // `MethodKind` mapping — a pure `FnDecl.task`/`.stream` → variant-name lookup,
        // tested directly against parsed `FnDecl`s rather than through a full struct-
        // method pipeline: `stream` methods cannot actually be declared inside a struct
        // body today (a genuine, pre-existing, Introspect-unrelated parser gap — struct-
        // body parsing has no `TokenKind::Stream` arm at all, unlike top-level function
        // declarations, which do support `stream`) — so there is no real Boring source
        // that would produce a `Method` entry with `kind: MethodKind::Stream` yet. This
        // parses each shape as a top-level function instead and feeds the resulting
        // `FnDecl` straight to the mapping function, to verify the mapping itself is
        // correct and ready for whenever that parser gap closes.
        let src = "\
def plainFn():\n    print \"p\"\n\n\
task def taskFn():\n    print \"t\"\n\n\
stream int streamFn():\n    yield 1\n";
        let tokens = crate::lexer::lex(src).expect("lex error");
        let program = crate::parser::parse(tokens).expect("parse error");
        let find = |name: &str| -> FnDecl {
            program.items.iter().find_map(|it| match it {
                Item::Fn(f) if f.name == name => Some(f.clone()),
                _ => None,
            }).unwrap_or_else(|| panic!("no fn '{}' found", name))
        };
        assert_eq!(Transpiler::introspect_method_kind_expr(&find("plainFn")), "MethodKind::Plain");
        assert_eq!(Transpiler::introspect_method_kind_expr(&find("taskFn")), "MethodKind::Task");
        assert_eq!(Transpiler::introspect_method_kind_expr(&find("streamFn")), "MethodKind::Stream");
    }

    #[test]
    fn introspect_prelude_emits_method_kind_enum_with_all_three_variants() {
        let src = "struct Point as Introspect:\n    float x\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("enum MethodKind { Plain, Task, Stream }"),
            "the MethodKind enum prelude must declare all three variants, got:\n{}", code);
        assert!(code.contains("kind: MethodKind,"),
            "Method's struct definition must carry the new `kind` field, got:\n{}", code);
        assert!(code.contains("isCallable: bool,"),
            "Method's struct definition must carry the new `isCallable` field, got:\n{}", code);
    }

    #[test]
    fn set_param_add_and_remove_use_hashset_methods() {
        let src = "\
def note(mut {int} s):\n    s.add(42)\n\n\
def drop(mut {int} s):\n    s.remove(1)\n\n\
def main():\n    mut {int} seen = {1, 2}\n    note(seen)\n    drop(seen)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("s.insert(42);"), "`mut {{int}}` param `.add` must be `.insert`, got:\n{}", code);
        assert!(code.contains("s.remove(&1);"), "`mut {{int}}` param `.remove(v)` must be `.remove(&v)`, got:\n{}", code);
        assert!(!code.contains(".add(42)"), "no raw `.add` may survive, got:\n{}", code);
        assert!(!code.contains("__boring_idx"), "set `.remove` must not be array remove-at, got:\n{}", code);
    }

    #[test]
    fn array_param_used_only_via_length_is_a_plain_vec_ref() {
        // A size query is neutral for the GPU-resident-arg scan: `.length`/`.count` alone must
        // not promote the parameter to `BoringKernelArg<T>` (undefined on non-GPU targets).
        let src = "\
def int count_seen([int] xs):\n    xs.length\n\n\
def int count_of([int] xs):\n    xs.count\n\n\
def main():\n    let [int] a = [5, 6, 7]\n    print count_seen(a)\n    print count_of(a)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn count_seen(xs: &Vec<isize>)"), "got:\n{}", code);
        assert!(code.contains("fn count_of(xs: &Vec<isize>)"), "got:\n{}", code);
        assert!(!code.contains("BoringKernelArg"), "no GPU arg type may appear, got:\n{}", code);
    }

    #[test]
    fn set_param_in_struct_method_uses_hashset_methods() {
        let src = "\
struct Bag:\n    var int total = 0\n\n    def fill(mut {int} s):\n        s.add(7)\n        s.remove(1)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("s.insert(7);"), "got:\n{}", code);
        assert!(code.contains("s.remove(&1);"), "got:\n{}", code);
    }

    #[test]
    fn mut_struct_param_of_free_fn_is_lent_mut_ref() {
        // A write through a `mut` struct param (field assign, compound assign, `def` call) is
        // what its universal `&mut T` borrow is for: the signature is `&mut T` and the call
        // site lends `&mut place` -- never the by-value `mut p: T` + `.clone()` whose mutation
        // landed on the throwaway clone.
        let src = r#"
struct Counter:
    var int value = 0

    def inc():
        value += 1

def set_value(mut Counter c):
    c.value = 5

def add_value(mut Counter c):
    c.value += 10

def call_inc(mut Counter c):
    c.inc()

def main():
    mut Counter c = Counter()
    set_value(c)
    add_value(c)
    call_inc(c)
    print c.value
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        for f in ["set_value", "add_value", "call_inc"] {
            assert!(code.contains(&format!("fn {}(mut c: &mut Counter)", f)), "{} signature, got:\n{}", f, code);
            assert!(code.contains(&format!("{}(&mut c);", f)), "{} call site, got:\n{}", f, code);
        }
        assert!(!code.contains("c.clone()"), "no argument may be cloned, got:\n{}", code);
    }

    #[test]
    fn plain_struct_param_stays_a_shared_borrow() {
        // Only `mut` lends mutably: a plain parameter stays a read-only `&T` borrow.
        let src = r#"
struct Counter:
    var int value = 0

def show(Counter c):
    print c.value

def main():
    mut Counter c = Counter()
    show(c)
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn show(c: &Counter)"), "got:\n{}", code);
        assert!(!code.contains("&mut Counter"), "got:\n{}", code);
    }

    #[test]
    fn single_uppercase_letter_struct_mut_param_is_lent() {
        // `struct P` -- a one-uppercase-letter name -- is parsed as `Type::TypeParam` (the
        // implicit-generic spelling) at its uses; it must still be recognised as the declared
        // struct it names and lent like any other.
        let src = r#"
struct P:
    var int x = 0

def poke(mut P p):
    p.x = 7

def main():
    var mut P p = P()
    poke(p)
    print p.x
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn poke(mut p: &mut P)"), "got:\n{}", code);
        assert!(code.contains("poke(&mut p);"), "got:\n{}", code);
    }

    #[test]
    fn mut_struct_param_of_method_is_lent_mut_ref() {
        // Universal borrow inference is excluded for methods, so the lend is unconditional there:
        // `&mut T` signature whatever the body does, `&mut place` at every call site (a local, a
        // bare struct field via implicit self on a *different* field's method -- lending a field
        // of `self` to a method of `self` itself is rustc's own E0499 -- and a forwarded lent
        // param of the caller).
        let src = r#"
struct Counter:
    var int value = 0

struct Bumper:
    var int hits = 0

    def bump(mut Counter c):
        c.value += 1
        hits += 1

    def forward(mut Counter c):
        self.bump(c)

    def noop(mut Counter c):
        hits += 1

struct Holder:
    var mut Counter inner = Counter()
    var mut Bumper b = Bumper()

    def bump_inner():
        b.bump(inner)

def main():
    mut Bumper h = Bumper()
    mut Counter c = Counter()
    h.bump(c)
    h.forward(c)
    h.noop(c)
    print c.value
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn bump(&mut self, c: &mut Counter)"), "got:\n{}", code);
        assert!(code.contains("fn noop(&mut self, c: &mut Counter)"), "body-independent, got:\n{}", code);
        assert!(code.contains("self.b.bump(&mut self.inner);"), "implicit-self field arg, got:\n{}", code);
        assert!(code.contains("self.bump(&mut *c);"), "forwarded lent param is reborrowed, got:\n{}", code);
        assert!(code.contains("h.bump(&mut c);"), "local arg, got:\n{}", code);
        assert!(code.contains("h.noop(&mut c);"), "got:\n{}", code);
        assert!(!code.contains("c.clone()"), "no argument may be cloned, got:\n{}", code);
    }

    #[test]
    fn mut_struct_param_of_trait_method_signature_matches_impl() {
        // A trait's header-only signature has no body to infer from: it must render `&mut T`
        // exactly like the implementing method, or the impl would not match the trait.
        let src = r#"
struct Counter:
    var int value = 0

trait Poker:
    def poke_it(mut Counter c)

struct A as Poker:
    def poke_it(mut Counter c):
        c.value += 100
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.matches("fn poke_it(&mut self, c: &mut Counter)").count() >= 2,
            "trait signature and impl must agree, got:\n{}", code);
    }

    #[test]
    fn mut_params_of_trait_typed_receiver_are_lent_from_the_trait_decl() {
        // A receiver typed as a bare trait (`Box<dyn Poker>`) has no struct method decl: the
        // lent flags come from the trait's own declaration, struct and collection params alike.
        let src = r#"
struct Counter:
    var int value = 0

trait Poker:
    def poke_it(mut Counter c)
    def stamp(mut [uint8] b)

def drive(mut Poker p, mut Counter c, mut [uint8] b):
    p.poke_it(c)
    p.stamp(b)
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("p.poke_it(&mut *c);"), "got:\n{}", code);
        assert!(code.contains("p.stamp(&mut *b);"), "got:\n{}", code);
        assert!(!code.contains("c.clone()") && !code.contains("b.clone()"), "got:\n{}", code);
    }

    #[test]
    fn var_struct_param_stays_the_rebind_out_param() {
        // `var` (rebindable) is a different capability from `mut`: it keeps its own out-param
        // path, for a free function and a method alike.
        let src = r#"
struct Counter:
    var int value = 0

struct Holder:
    var int hits = 0

    def reseat(var Counter c):
        c = Counter()

def reset(var Counter c):
    c = Counter()

def main():
    var Counter c = Counter()
    var Holder h = Holder()
    reset(c)
    h.reseat(c)
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("reset(&mut c);"), "got:\n{}", code);
        assert!(code.contains("h.reseat(&mut c);"), "got:\n{}", code);
    }

    #[test]
    fn set_local_does_not_leak_into_next_function_array_param() {
        // `seen` is a set local in `first`, but a plain array parameter in `second`: the
        // stale `set_vars` entry used to make `second`'s `seen.remove(0)` a `HashSet` remove.
        let src = "\
def first():\n    mut {int} seen = {1}\n    seen.add(2)\n\n\
def second(mut [int] seen):\n    seen.remove(0)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("seen.insert(2);"), "got:\n{}", code);
        assert!(!code.contains("seen.remove(&0)"), "array param must keep array remove-at, got:\n{}", code);
    }

    #[test]
    fn bare_sibling_method_call_is_a_boring_diagnostic() {
        let src = "\
struct Src:\n    var int pos = 0\n\n    def one(int n):\n        pos += n\n\n    def two(int n):\n        one(n)\n";
        let errors = transpile_src_errors(src);
        assert!(errors.contains("cannot call method 'one' without a receiver inside struct 'Src'"), "got: {}", errors);
        assert!(errors.contains("self.one(...)"), "diagnostic must name the fix, got: {}", errors);
    }

    #[test]
    fn explicit_self_sibling_method_call_is_accepted() {
        let src = "\
struct Src:\n    var int pos = 0\n\n    def one(int n):\n        pos += n\n\n    def two(int n):\n        self.one(n)\n";
        assert_eq!(transpile_src_errors(src), "");
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("self.one("), "got:\n{}", code);
    }

    #[test]
    fn bare_call_to_same_named_top_level_fn_inside_method_is_not_flagged() {
        let src = "\
def int one(int n):\n    n\n\n\
struct Src:\n    var int pos = 0\n\n    def one(int n):\n        pos += n\n\n    def two(int n):\n        let x = one(n)\n        pos += x\n";
        assert_eq!(transpile_src_errors(src), "");
    }

    // ── `mut` parameters lent through receivers whose type needs a declaration walk ──────
    // `tests/cases/mut_param_unresolved_receiver.br` covers the same through a real `cargo run`.

    /// A `Counter`/`Shape` prelude: `Shape.tap` takes a `mut Counter`, `Shape.mark` a `mut [uint8]`
    /// (`req`, so loop variables / non-`mut` elements may call them), and a `Poker` trait with `req`
    /// twins implemented by `Sq`.
    const LENT_PRELUDE: &str = "\
struct Counter:\n    var int value = 0\n\n    def inc(int by):\n        value += by\n\n\
struct Shape:\n    var int n = 0\n\n    req tap(mut Counter c):\n        c.inc(1)\n\n    req mark(mut [uint8] b):\n        b[0] += 1 as uint8\n\n\
trait Poker:\n    req tap(mut Counter c)\n\n\
struct Sq as Poker:\n    var int n = 0\n\n    req tap(mut Counter c):\n        c.inc(2)\n\n\
def Shape make():\n    Shape()\n\n";

    fn lent_case(main_body: &str) -> (String, String) {
        let src = format!("{}def main():\n    mut Counter c = Counter()\n    mut [uint8] b = [0 as uint8 for i in 0..<4]\n{}", LENT_PRELUDE, main_body);
        let errs = transpile_src_errors(&src);
        (transpile_src_with_config(&src, TranspileConfig::default()), errs)
    }

    #[test]
    fn mut_args_are_lent_to_an_indexed_mut_element_receiver() {
        // `[mut Shape]`: the element type carries `mut`, which must not hide the struct.
        let (code, errs) = lent_case("    mut [mut Shape] ms = [Shape()]\n    ms[0].tap(c)\n    ms[0].mark(b)\n");
        assert_eq!(errs, "");
        assert!(code.contains("].tap(&mut c)") && code.contains("].mark(&mut b)"), "got:\n{}", code);
    }

    #[test]
    fn mut_args_are_lent_to_trait_array_element_and_loop_receivers() {
        let (code, errs) = lent_case("    let [Poker] ps = [Sq()]\n    ps[0].tap(c)\n    for p in ps:\n        p.tap(c)\n");
        assert_eq!(errs, "");
        assert_eq!(code.matches(".tap(&mut c)").count(), 2, "got:\n{}", code);
    }

    #[test]
    fn mut_args_are_lent_to_call_and_dict_value_receivers() {
        let (code, errs) = lent_case("    make().tap(c)\n    mut {string=mut Shape} d = {\"a\" = Shape()}\n    d[\"a\"].mark(b)\n    for k, v in d:\n        v.tap(c)\n");
        assert_eq!(errs, "");
        assert!(code.contains("make().tap(&mut c)"), "got:\n{}", code);
        assert!(code.contains(".mark(&mut b)"), "got:\n{}", code);
        assert!(code.contains("v.tap(&mut c)"), "got:\n{}", code);
    }

    #[test]
    fn mut_args_are_lent_to_typed_and_collection_hinted_closure_param_receivers() {
        let (code, errs) = lent_case("    let [Shape] ss = [Shape()]\n    var f = (Shape s): s.tap(c)\n    f(Shape())\n    _ = ss.map((s): s.tap(c))\n");
        assert_eq!(errs, "");
        assert_eq!(code.matches("s.tap(&mut c)").count(), 2, "got:\n{}", code);
    }

    #[test]
    fn mut_args_are_lent_to_trait_bounded_generic_receiver() {
        let src = format!("{}def run<T as Poker>(mut T p):\n    mut Counter c = Counter()\n    p.tap(c)\n", LENT_PRELUDE);
        assert_eq!(transpile_src_errors(&src), "");
        let code = transpile_src_with_config(&src, TranspileConfig::default());
        assert!(code.contains("p.tap(&mut c)"), "got:\n{}", code);
    }

    #[test]
    fn mut_args_are_lent_through_an_actor_receiver() {
        let (code, errs) = lent_case("    mut Shape'actor ac = Shape()\n    ac.tap(c)\n");
        assert_eq!(errs, "");
        assert!(code.contains(".tap(&mut c)"), "got:\n{}", code);
        assert!(!code.contains("tap(c.clone())"), "got:\n{}", code);
    }

    #[test]
    fn untyped_closure_param_receiver_with_lent_arg_is_a_boring_diagnostic() {
        // `s` has no type anywhere: which `tap` (and which of its parameters is `mut`) is
        // unknowable, so a by-value argument would be invalid Rust -- say so instead.
        let (_, errs) = lent_case("    let g = (s): s.tap(c)\n");
        assert!(errs.contains("cannot lend `c` to the `mut` parameter of `.tap()`"), "got: {}", errs);
        assert!(errs.contains("declare its type"), "diagnostic must name the fix, got: {}", errs);
    }

    #[test]
    fn untyped_receiver_with_non_lent_method_args_is_not_flagged() {
        // `inc`'s parameter is a plain int: nothing to lend, so no diagnostic however unknown the receiver.
        let (_, errs) = lent_case("    let g = (x): x.inc(1)\n");
        assert_eq!(errs, "");
    }

    #[test]
    fn test_data_field_never_infers_lock_qualifier() {
        let src = r#"
data Inner:
    int count = 0
    def inc():
        count += 1

data Outer:
    Inner inner

    def bump():
        inner.inc()
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        // inner field must stay pure value (pub inner: Inner), never Arc<Mutex<Inner>> or RefCell
        assert!(code.contains("pub inner: Inner"), "got:\n{}", code);
        assert!(!code.contains("Mutex<Inner>"), "got:\n{}", code);
    }

    #[test]
    fn test_custom_ecs_query_transpile() {
        let src = r#"
@component
struct Ball:
    int id

@component
struct Velocity:
    float32 x
    float32 y

@component
struct Transform:
    float32 x

@component
struct Brick:
    int hp

@component
struct Collider:
    int id

pub def check_collisions(mut commands) throws:
    mut ball = single (mut Velocity, Transform) where Ball
    for entity, Transform collider_trans, Brick? maybe_brick where Collider:
        if maybe_brick != nil:
            commands.entity(entity).despawn()
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("pub fn check_collisions("), "got:\n{}", code);
        assert!(code.contains("ball_query: Query<(&mut Velocity, &Transform), With<Ball>>"), "got:\n{}", code);
        assert!(code.contains("collider_query: Query<(Entity, &Transform, Option<&Brick>), With<Collider>>"), "got:\n{}", code);
        assert!(code.contains("-> Result<(), Box<dyn std::error::Error + Send + Sync>>"), "got:\n{}", code);
        assert!(code.contains("ball_query.get_single().map_err("), "got:\n{}", code);
        assert!(code.contains("for (entity, collider_trans, maybe_brick) in &collider_query"), "got:\n{}", code);
        assert!(code.contains("Ok(())"), "got:\n{}", code);
    }

    #[test]
    fn test_query_fetch_expressions() {
        let src = r#"
@component
struct Ball:
    int id

@component
struct Transform:
    float32 x

@component
struct Brick:
    int hp

pub def ecs_system() throws:
    let ball = query single (Transform) where Ball
    let maybe_bonus = query single? (Transform) where Ball
    let has_ball = query any where Ball
    let bricks = query (Entity, Transform) where Brick
    let all_bricks = query all (Entity, Transform) where Brick
    if bricks.isEmpty():
        return
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("pub fn ecs_system("), "got:\n{}", code);
        assert!(code.contains("ball_query: Query<&Transform, With<Ball>>"), "got:\n{}", code);
        assert!(code.contains("ball_query.get_single().map_err("), "got:\n{}", code);
        assert!(code.contains("ball_query_1.iter().next()"), "got:\n{}", code);
        assert!(code.contains("!ball_query_2.is_empty()"), "got:\n{}", code);
        assert!(code.contains("brick_query: Query<(Entity, &Transform), With<Brick>>"), "got:\n{}", code);
        assert!(code.contains("(&brick_query)"), "got:\n{}", code);
        assert!(code.contains("brick_query_1: Query<(Entity, &Transform), With<Brick>>"), "got:\n{}", code);
        assert!(code.contains("(&brick_query_1)"), "got:\n{}", code);
    }

    #[test]
    fn test_ambient_and_qualified_ecs_access() {
        let src = r#"
@resource
var int score = 0

@component
struct Paddle:
    int id

pub def move_paddle(mut Transform transform) where Paddle:
    let dt = time.delta as float32
    let dt2 = ecs.time.delta as float32
    if input.pressed(KeyCode.ArrowLeft):
        transform.x -= 100.0 * dt
    if ecs.input.pressed(KeyCode.ArrowRight):
        transform.x += 100.0 * dt2
    commands.spawn(Paddle(id = 1))
    ecs.commands.spawn(Paddle(id = 2))
    resource.score += 10
    ecs.resource.score += 20
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("pub fn move_paddle("), "got:\n{}", code);
        assert!(code.contains("time: Res<Time>"), "got:\n{}", code);
        assert!(code.contains("input: Res<ButtonInput<KeyCode>>"), "got:\n{}", code);
        assert!(code.contains("mut commands: Commands"), "got:\n{}", code);
        assert!(code.contains("mut res_score: ResMut<Res_score>"), "got:\n{}", code);
        assert!(code.contains("time.delta_secs()"), "got:\n{}", code);
        assert!(code.contains("input.pressed("), "got:\n{}", code);
        assert!(code.contains("commands.spawn("), "got:\n{}", code);
        assert!(code.contains("res_score.0 += 10"), "got:\n{}", code);
        assert!(code.contains("res_score.0 += 20"), "got:\n{}", code);
    }

    #[test]
    fn test_ambient_and_qualified_entity_and_despawn() {
        let src = r#"
@component
data Enemy:
    bool is_dead

@component
data Ghost:
    bool vanished

def kill_enemy(Enemy e) where Dead:
    let my_id = entity
    despawn()

def kill_ghost(Ghost g) where Vanished:
    let gid = ecs.entity
    ecs.despawn()
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("pub fn kill_enemy("), "got:\n{}", code);
        assert!(code.contains("mut commands: Commands"), "got:\n{}", code);
        assert!(code.contains("(__entity, e)"), "got:\n{}", code);
        assert!(code.contains("let my_id = __entity;"), "got:\n{}", code);
        assert!(code.contains("commands.entity(__entity).despawn()"), "got:\n{}", code);

        assert!(code.contains("pub fn kill_ghost("), "got:\n{}", code);
        assert!(code.contains("(__entity, g)"), "got:\n{}", code);
        assert!(code.contains("let gid = __entity"), "got:\n{}", code);
    }

    #[test]
    fn test_camel_case_input_and_time_methods() {
        let src = r#"
@component
data Player:
    int id

def handle_player_input(Player p):
    if input.justPressed(KeyCode.Space):
        let dt = time.deltaSecs()
        let total = time.elapsedSecs()
    if ecs.input.justReleased(KeyCode.Escape):
        let dt2 = ecs.time.delta_secs()
    if input.just_pressed(KeyCode.Enter):
        let dt3 = time.deltaSecs
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("pub fn handle_player_input("), "got:\n{}", code);
        assert!(code.contains("input: Res<ButtonInput<KeyCode>>"), "got:\n{}", code);
        assert!(code.contains("time: Res<Time>"), "got:\n{}", code);
        assert!(code.contains("input.just_pressed(KeyCode::Space)"), "got:\n{}", code);
        assert!(code.contains("let dt = (time.delta_secs() as f64);"), "got:\n{}", code);
        assert!(code.contains("let total = (time.elapsed_secs_f64());"), "got:\n{}", code);
        assert!(code.contains("input.just_released(KeyCode::Escape)"), "got:\n{}", code);
        assert!(code.contains("let dt2 = (time.delta_secs() as f64);"), "got:\n{}", code);
        assert!(code.contains("input.just_pressed(KeyCode::Enter)"), "got:\n{}", code);
        assert!(code.contains("let dt3 = (time.delta_secs() as f64)"), "got:\n{}", code);
    }

    #[test]
    fn test_for_where_and_collection_query_transpile() {
        let src = r#"
def test():
    let nums = [1, 2, 3, 4, 5]
    for x in nums where x > 2:
        print "{x}"

    let q1 = query x in nums where x % 2 == 0
    let q2 = query x in nums where x % 2 == 0 |> x * 10
    let first_match = query first? x in nums where x == 3
    let single_match = query single? x in nums where x == 4
    let any_match = query any? x in nums where x > 2
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("if !((x > 2)) { continue; }"), "got:\n{}", code);
        assert!(code.contains(".into_iter().filter(|x| ((x % 2) == 0))"), "got:\n{}", code);
        assert!(code.contains(".into_iter().filter(|x| ((x % 2) == 0)).map(|x| (x * 10))"), "got:\n{}", code);
        assert!(code.contains(".into_iter().filter(|x| (x == 3)).next()"), "got:\n{}", code);
        assert!(code.contains("query single"), "got:\n{}", code);
        assert!(code.contains(".into_iter().filter(|x| (x > 2)).next()"), "got:\n{}", code);
    }

    #[test]
    fn test_for_where_and_collection_query_interpreter() {
        let src = r#"
var sum = 0
for x in [1, 2, 3, 4, 5, 6] where x % 2 == 0:
    sum = sum + x

let evens = query x in [1, 2, 3, 4] where x % 2 == 0
let doubled = query x in [1, 2, 3, 4] where x > 2 |> x * 10
let collected = (query x in [1, 2, 3] |> x + 1).collect()
let found = query first? x in [10, 20, 30] where x > 15
let not_found = query first? x in [10, 20, 30] where x > 100
let s_found = query single? x in [10, 20, 30] where x == 20
let s_not_found = query single? x in [10, 20, 30] where x > 50
let a_found = query any? x in [10, 20, 30] where x > 15
"#;
        let tokens = crate::lexer::lex(src).expect("lex error");
        let program = crate::parser::parse(tokens).expect("parse error");
        let mut interp = crate::interpreter::Interpreter::new();
        interp.exec_program(&program).expect("exec error");

        assert_eq!(interp.global.borrow().get("sum"), Some(crate::interpreter::Value::Int(12)));
        assert_eq!(
            interp.global.borrow().get("evens"),
            Some(crate::interpreter::Value::Array(std::rc::Rc::new(vec![
                crate::interpreter::Value::Int(2),
                crate::interpreter::Value::Int(4),
            ])))
        );
        assert_eq!(
            interp.global.borrow().get("doubled"),
            Some(crate::interpreter::Value::Array(std::rc::Rc::new(vec![
                crate::interpreter::Value::Int(30),
                crate::interpreter::Value::Int(40),
            ])))
        );
        assert_eq!(
            interp.global.borrow().get("collected"),
            Some(crate::interpreter::Value::Array(std::rc::Rc::new(vec![
                crate::interpreter::Value::Int(2),
                crate::interpreter::Value::Int(3),
                crate::interpreter::Value::Int(4),
            ])))
        );
        assert_eq!(interp.global.borrow().get("found"), Some(crate::interpreter::Value::Int(20)));
        assert_eq!(interp.global.borrow().get("not_found"), Some(crate::interpreter::Value::Nil));
        assert_eq!(interp.global.borrow().get("s_found"), Some(crate::interpreter::Value::Int(20)));
        assert_eq!(interp.global.borrow().get("s_not_found"), Some(crate::interpreter::Value::Nil));
        assert_eq!(interp.global.borrow().get("a_found"), Some(crate::interpreter::Value::Int(20)));

        // Test error on multiple elements with `query single`
        let src_err = r#"let bad = query single x in [1, 2, 3] where x > 1"#;
        let p_err = crate::parser::parse(crate::lexer::lex(src_err).unwrap()).unwrap();
        let mut interp_err = crate::interpreter::Interpreter::new();
        assert!(interp_err.exec_program(&p_err).is_err());

        // Test error on missing element with non-optional `query first`
        let src_err2 = r#"let bad = query first x in [1, 2, 3] where x > 10"#;
        let p_err2 = crate::parser::parse(crate::lexer::lex(src_err2).unwrap()).unwrap();
        let mut interp_err2 = crate::interpreter::Interpreter::new();
        assert!(interp_err2.exec_program(&p_err2).is_err());
    }

    #[test]
    fn test_unannotated_mut_from_fn_call_captured_in_closure() {
        let src = r#"
struct State:
    int count = 0
    def inc():
        count += 1

def State makeState():
    State(count = 0)

def mutate(mut State s):
    s.inc()

def run():
    mut st = makeState()
    mutate(st)
    let f = ():
        mutate(st)
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("Arc<std::sync::Mutex<State>>"), "expected Arc<Mutex<State>>, got:\n{}", code);
        assert!(code.contains("Arc::new(std::sync::Mutex::new(makeState()))"), "expected Arc::new(Mutex::new(makeState())), got:\n{}", code);
        assert!(code.contains("mutate(&mut *st.lock().unwrap())"), "expected mutate(&mut *st.lock().unwrap()), got:\n{}", code);
    }

    #[test]
    fn test_already_qualified_actor_field_and_return_not_double_wrapped() {
        let src = r#"
struct Session:
    var int id = 0

struct Server:
    var Session'actor session

def Session'actor make_session():
    Session(id = 2)

def test(Server'actor srv):
    let s = Server(session = Session(id = 1))
    let cur = s.session
    let parent = srv.session
    let made = make_session()
    print cur.id
    print parent.id
    print made.id
"#;
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(
            code.contains("let cur: Arc<std::sync::Mutex<Session>> = Arc::clone(&s.session);"),
            "expected Arc::clone(&s.session) for cur, got:\n{}",
            code
        );
        assert!(
            code.contains("Arc::clone(&srv.lock().unwrap().session)"),
            "expected Arc::clone(&srv.lock().unwrap().session) for parent, got:\n{}",
            code
        );
        assert!(
            !code.contains("Arc::new(std::sync::Mutex::new(Arc::clone("),
            "must not double-wrap Arc::clone in Arc<Mutex<...>>, got:\n{}",
            code
        );
        assert!(
            !code.contains("Arc::new(std::sync::Mutex::new(make_session("),
            "must not double-wrap already-actor fn return in Arc<Mutex<...>>, got:\n{}",
            code
        );
    }
