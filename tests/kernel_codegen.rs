// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Kernel (Rust-for-Linux, no_std) codegen snapshot tests.
//
// These tests verify the text emitted by `boring build --linux`
// without requiring the real Linux kernel build system. Each test:
//   1. Writes a Boring source snippet to a temp file.
//   2. Invokes `boring build --linux <file>`.
//   3. Reads the generated src/lib.rs.
//   4. Asserts that the generated text contains the expected patterns.
//
// Run with:
//   cargo test --test kernel_codegen

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Invoke `boring build --linux <file>` and return the generated
/// src/lib.rs text.
///
/// boring names the output directory `<stem>_linux` next to the source
/// file, so we place the source in a dedicated temp dir and read from there.
fn kernel_codegen(test_name: &str, src: &str) -> String {
    let bin = env!("CARGO_BIN_EXE_boring");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("kernel_codegen").join(test_name);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // Source file named "test.br" → boring creates "test_linux/" next to it.
    let br_file    = tmp.join("test.br");
    let kernel_dir = tmp.join("test_linux");
    fs::write(&br_file, src).unwrap();

    let result = Command::new(bin)
        .args(["build", "--linux"])
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("[{test_name}] failed to invoke boring: {e}"));

    assert!(
        result.status.success(),
        "[{test_name}] boring build --linux failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );

    fs::read_to_string(kernel_dir.join("src/lib.rs")).unwrap_or_default()
}

#[test]
fn linux_backend_is_a_standalone_flag_not_a_compute_target() {
    let bin = env!("CARGO_BIN_EXE_boring");

    for obsolete_target in ["kernel", "linux"] {
        let result = Command::new(bin)
            .args(["build", "--target", obsolete_target])
            .output()
            .unwrap_or_else(|e| panic!("failed to invoke boring: {e}"));

        assert!(
            !result.status.success(),
            "--target {obsolete_target} must not remain accepted"
        );
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr.contains(&format!("unknown target '{obsolete_target}'")),
            "expected an unknown-target diagnostic for --target {obsolete_target}; got:\n{stderr}"
        );
        assert!(
            stderr.contains("use --linux for Rust-for-Linux"),
            "expected the diagnostic to point to --linux; got:\n{stderr}"
        );
    }
}

// ─── while-let self-shorthand loop ──────────────────────────────────────────
//
// Regression for the gap found while porting ecb2690/3a42aa1 (the
// standard-target `while let v:` self-shorthand fix) to the Linux target:
// the kernel transpiler's `Stmt::WhileLet` arm had none of the standard
// transpiler's self-shorthand handling, so it would naively emit
// `while let Some(v) = v { ...; v = next; }` — shadow-colliding the outer
// `Option`-typed `v` with the loop-pattern's own unwrapped `v`.

#[test]
fn while_let_self_shorthand_reassignment_uses_distinct_outer_binding() {
    let rs = kernel_codegen("self_shorthand_reassign", r#"
def int? next_val(int c):
    if c <= 0: return nil
    return c - 1

def void run():
    var int? countdown = next_val(3)
    while let countdown:
        countdown = next_val(countdown)
"#);
    // The outer binding must get a distinct mangled name — the loop-pattern's
    // own unwrapped binding keeps the original name unshadowed-by-codegen.
    assert!(rs.contains("let mut __wl_countdown = countdown;"),
        "expected outer binding to be renamed to __wl_countdown;\ngot:\n{rs}");
    assert!(rs.contains("while let Some(countdown) = __wl_countdown {"),
        "expected loop head to match on the renamed outer binding;\ngot:\n{rs}");
    // The body's reassignment must target the renamed outer binding, not the
    // loop-pattern's own (shadowed) unwrapped `countdown`.
    assert!(rs.contains("__wl_countdown = next_val(countdown);"),
        "expected body reassignment to be redirected to __wl_countdown;\ngot:\n{rs}");
    // Post-loop sync writes the final (`None`) state back to the real binding.
    assert!(rs.contains("countdown = __wl_countdown;"),
        "expected post-loop sync back into countdown;\ngot:\n{rs}");
}

#[test]
fn while_let_self_shorthand_break_refills_outer_binding() {
    let rs = kernel_codegen("self_shorthand_break", r#"
def int? next_val(int c):
    if c <= 0: return nil
    return c - 1

def void run():
    var int? countdown = next_val(5)
    while let countdown:
        if countdown == 2:
            break
        countdown = next_val(countdown)
"#);
    // Reaching `break` must refill the outer binding from the loop-local
    // unwrapped value first, or the post-loop sync reads it back out while
    // still in the moved-from state the `while let Some(name) = outer` match
    // left it in on that iteration (Rust E0382).
    assert!(rs.contains("__wl_countdown = Some(countdown);\n") || rs.contains("__wl_countdown = Some(countdown);"),
        "expected break to refill __wl_countdown before exiting;\ngot:\n{rs}");
    let break_pos = rs.find("break;").expect("expected a break statement");
    let refill_pos = rs.find("__wl_countdown = Some(countdown);").expect("expected refill before break");
    assert!(refill_pos < break_pos,
        "expected the outer-binding refill to precede `break;`;\ngot:\n{rs}");
    assert!(rs.contains("countdown = __wl_countdown;"),
        "expected post-loop sync back into countdown;\ngot:\n{rs}");
}

#[test]
fn while_let_self_shorthand_nested_loop_break_is_unaffected() {
    let rs = kernel_codegen("self_shorthand_nested_break", r#"
def int? next_val(int c):
    if c <= 0: return nil
    return c - 1

def void run():
    var int? countdown = next_val(5)
    while let countdown:
        for i in 0..<10:
            if i == 3:
                break
        countdown = next_val(countdown)
"#);
    // A `break` belonging to the nested `for` loop must never be mistaken
    // for exiting the enclosing while-let-shorthand loop — no refill of the
    // outer binding should be emitted around it.
    let for_pos = rs.find("for i in").expect("expected a for loop");
    let after_for = &rs[for_pos..];
    assert!(!after_for.contains("__wl_countdown = Some(countdown);"),
        "the nested for-loop's break must not trigger the outer-binding refill;\ngot:\n{rs}");
}

// ── Linux target: float detection, `catch`, `'actor'task` / `'guard'task` ────

/// Run `boring build --linux` on `src` and return (success, stderr).
fn kernel_build_result(test_name: &str, src: &str) -> (bool, String) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("kernel_codegen").join(test_name);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br_file = tmp.join("test.br");
    fs::write(&br_file, src).unwrap();
    let out = Command::new(bin).args(["build", "--linux"]).arg(&br_file).output().unwrap();
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn float_origins_without_literal_or_declared_type_are_rejected() {
    // Each of these yields a float purely by inference — no float literal, no float type.
    for (i, expr) in [
        "float(n)", "float32(n)", "PI", "E", "INF", "NAN", "bitsToFloat(n)",
        "\"1.5\".parseFloat()", "n.toFloat()",
    ].iter().enumerate() {
        let src = format!("def int g(int n):\n    let x = {expr}\n    return 1\n");
        let (ok, err) = kernel_build_result(&format!("float_origin_{i}"), &src);
        assert!(!ok, "`{expr}` must be rejected on --linux");
        assert!(err.contains("FPU is disabled"), "`{expr}`: unexpected diagnostic:\n{err}");
    }
}

#[test]
fn integer_code_and_user_bound_float_constant_names_still_pass() {
    // A user binding named like a float constant shadows it; plain int math is untouched.
    let rs = kernel_codegen("float_shadow_ok", "def int g(int n):\n    let E = 5\n    let y = abs(n) + E\n    return y\n");
    assert!(rs.contains("fn g"), "got:\n{rs}");
}

#[test]
fn catch_clauses_are_rejected_not_silently_dropped() {
    let (ok, err) = kernel_build_result("catch_rejected", r#"
def int f() throws:
    return 1

def int g():
    try:
        let a = f()
    catch String:
        return 0
    return 1
"#);
    assert!(!ok, "`catch` must be diagnosed on --linux");
    assert!(err.contains("`catch String` is not supported on the Rust-for-Linux Linux target"), "got:\n{err}");
}

#[test]
fn actor_task_and_guard_task_use_kernel_locks_not_a_borrow() {
    let rs = kernel_codegen("actor_guard_task", r#"
struct C:
    var int n = 0

struct H:
    C'actor'task a
    C'guard'task b
"#);
    assert!(rs.contains("pub a: Arc<kernel::sync::Mutex<C>>,"), "got:\n{rs}");
    assert!(rs.contains("pub b: Arc<kernel::sync::RwLock<C>>,"), "got:\n{rs}");
}

// ── Linux target: fixed-point types (integer-only, no FPU) ───────────────────

const FIXED_SRC: &str = "def int32 g(int n):\n    let a = 3 as fixed16_16\n    let b = a * a\n    let c = b.sqrt()\n    return c.raw\n";

#[test]
fn fixed_point_emits_integer_newtypes_and_no_float_items() {
    let rs = kernel_codegen("fixed_basic", FIXED_SRC);
    assert!(rs.contains("pub struct Fixed16_16(pub i32)"), "got:\n{rs}");
    assert!(rs.contains("Fixed16_16::from_raw(196608)"), "literal cast must fold:\n{rs}");
    assert!(rs.contains(".sqrt()") && rs.contains(".raw()"), "got:\n{rs}");
    assert!(!rs.contains("3 as fixed16_16"), "got:\n{rs}");
    for banned in ["f64", "f32"] {
        assert!(!rs.contains(banned), "generated kernel code must contain no `{banned}`:\n{rs}");
    }
}

#[test]
fn fixed_point_members_fields_and_decimal_cast_literal() {
    let src = "struct Pt:\n    var fixed16_16 x = fixed16_16.fromRaw(65536)\n\n\
def int32 g(Pt p, fixed16_16 y, int n):\n    let h = 1.5 as fixed16_16\n    let z = (p.x + y) * h + 2\n    let w = n as fixed16_16\n    let s = z.sin() + z.cos() + z.exp() + w.ln()\n    let k = z.checkedAdd(y) else 0\n    return (s + k).raw + (z as int32) + fixed16_16.epsilon.raw\n";
    let rs = kernel_codegen("fixed_members", src);
    assert!(rs.contains("Fixed16_16::from_raw(98304)"), "1.5 folds to raw 98304:\n{rs}");
    assert!(rs.contains("Fixed16_16::from_raw(131072)"), "bare int literal operand takes the format:\n{rs}");
    assert!(rs.contains("__fx_cast_sat::<Fixed16_16, _>(n)"), "got:\n{rs}");
    assert!(rs.contains(".unwrap_or(Fixed16_16::from_raw(0))"), "got:\n{rs}");
    assert!(rs.contains("__FixedFormat::__to_i128(z) as i32"), "got:\n{rs}");
    assert!(rs.contains("Fixed16_16::EPSILON.raw()"), "got:\n{rs}");
    assert!(!rs.contains("f64") && !rs.contains("f32"), "got:\n{rs}");
}

#[test]
fn fixed_point_float_conversions_are_rejected() {
    for (i, body) in [
        "let x = 1.5\n    let a = x as fixed16_16\n    return 1",
        "let a = 3 as fixed16_16\n    let x = a * 1.5\n    return 1",
        "let a = 3 as fixed16_16\n    let x = a as float\n    return 1",
        "let a = 3 as fixed16_16\n    let x = a.toFloat()\n    return 1",
    ].iter().enumerate() {
        let src = format!("def int g(int n):\n    {body}\n");
        let (ok, err) = kernel_build_result(&format!("fixed_float_{i}"), &src);
        assert!(!ok, "case {i} must be rejected on --linux:\n{body}");
        assert!(err.contains("FPU is disabled"), "case {i}: unexpected diagnostic:\n{err}");
    }
}

/// Compiles the generated fixed-point module with plain `rustc` (`#![no_std]` kept, the
/// `kernel` crate import dropped — fixed-point code needs none) and, in a second build with a
/// std `main`, runs it: `g(0)` = raw of sqrt(3*3) = 3.0 in 16.16 = 196608.
#[test]
fn fixed_point_generated_code_compiles_no_std_and_computes() {
    let rs = kernel_codegen("fixed_compile", FIXED_SRC);
    let rs = rs.replace("extern crate kernel;", "");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("kernel_codegen").join("fixed_compile");
    // 1. no_std library build
    let lib = tmp.join("lib_check.rs");
    fs::write(&lib, &rs).unwrap();
    let out = Command::new("rustc").args(["--edition", "2021", "--crate-type", "lib", "--emit", "metadata", "-o"])
        .arg(tmp.join("lib_check.rmeta")).arg(&lib).output().expect("rustc not found");
    assert!(out.status.success(), "no_std build failed:\n{}\n--- code ---\n{rs}", String::from_utf8_lossy(&out.stderr));
    // 2. std build + run
    let main_rs = format!("{}\nfn main() {{ println!(\"{{}}\", g(0)); }}\n", rs.replace("#![no_std]", ""));
    let main_file = tmp.join("run_check.rs");
    fs::write(&main_file, main_rs).unwrap();
    let exe = tmp.join("run_check");
    let out = Command::new("rustc").args(["--edition", "2021", "-o"]).arg(&exe).arg(&main_file).output().unwrap();
    assert!(out.status.success(), "std build failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let run = Command::new(&exe).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "196608");
}

// ── Linux target: fixed-point inference scope, literal folding, panic-free policy ──

const FIXED_FOLD_SRC: &str = "\
def fixed16_16 one():
    return 1

def fixed16_16 twice(fixed16_16 x):
    return x * 2

def int32 f_fold():
    return twice(3).raw

def int32 f_wrap():
    return (fixed16_16.max + one()).raw

def int32 f_div0():
    let z = 0 as fixed16_16
    return (one() / z).raw

def int32 f_rem0():
    let z = 0 as fixed16_16
    return (one() % z).raw

def int32 f_sqrt_neg():
    let m = 0 - one()
    return m.sqrt().raw

def int32 f_ln0():
    let z = 0 as fixed16_16
    return z.ln().raw

def int32 f_cast_hi(int n):
    return (n as fixed16_16).raw

def int32 f_cast_lo(int n):
    return ((0 - n) as fixed16_16).raw

def int64 f_wide():
    let a = 3 as fixed64_64
    let b = a * a / a
    return (b as int64) + (a.raw as int64)
";

#[test]
fn fixed_point_literals_fold_in_return_arguments_and_assignments() {
    let src = "\
struct Acc:
    var fixed16_16 v

    req fixed16_16 get():
        return v

def fixed16_16 one():
    return 1

def fixed16_16 inc(fixed16_16 x, fixed16_16 by):
    return x + by

def int32 g(Acc a, int a2):
    var t = inc(1, 2)
    t = 5
    let k = a.get() + 1
    let u = Acc(7)
    return (t + k).raw + (a2 + 1)

def int32 h(int t):
    return t + 1
";
    let rs = kernel_codegen("fixed_fold", src);
    assert!(rs.contains("-> Fixed16_16 {\n    return Fixed16_16::from_raw(65536);"), "return literal:\n{rs}");
    assert!(rs.contains("inc(Fixed16_16::from_raw(65536), Fixed16_16::from_raw(131072))"), "call args:\n{rs}");
    assert!(rs.contains("t = Fixed16_16::from_raw(327680)"), "assignment:\n{rs}");
    assert!(rs.contains("(a.get() + Fixed16_16::from_raw(65536))"), "method return type tracked:\n{rs}");
    assert!(rs.contains("Acc(Fixed16_16::from_raw(458752))"), "constructor argument:\n{rs}");
    assert!(rs.contains("(a2 + 1)"), "an int binding is not folded:\n{rs}");
    // `t` is fixed in g but an int parameter in h: the per-function scope must not leak.
    assert!(rs.contains("(t + 1)"), "binding scope leaked across functions:\n{rs}");
}

#[test]
fn fixed_point_prelude_is_panic_free_and_semantics_are_total() {
    let rs = kernel_codegen("fixed_total", FIXED_FOLD_SRC);
    let prelude = &rs[rs.find("// ---- fixed-point support").expect("prelude")..];
    for banned in ["panic!", ".expect(", "assert!", "debug_assertions", ".unwrap()", "f64", "f32"] {
        assert!(!prelude.contains(banned), "kernel prelude must not contain `{banned}`");
    }
    let rs = rs.replace("extern crate kernel;", "");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("kernel_codegen").join("fixed_total");
    let lib = tmp.join("lib_check.rs");
    fs::write(&lib, &rs).unwrap();
    let out = Command::new("rustc").args(["--edition", "2021", "--crate-type", "lib", "--emit", "metadata", "-o"])
        .arg(tmp.join("lib_check.rmeta")).arg(&lib).output().expect("rustc not found");
    assert!(out.status.success(), "no_std build failed:\n{}\n{rs}", String::from_utf8_lossy(&out.stderr));
    // Run in a *debug* std build: nothing may panic, whatever the overflow checks.
    let main_rs = format!("{}\nfn main() {{ println!(\"{{}} {{}} {{}} {{}} {{}} {{}} {{}} {{}} {{}}\", f_fold(), f_wrap(), f_div0(), f_rem0(), f_sqrt_neg(), f_ln0(), f_cast_hi(70000), f_cast_lo(70000), f_wide()); }}\n",
        rs.replace("#![no_std]", ""));
    let main_file = tmp.join("run_check.rs");
    fs::write(&main_file, main_rs).unwrap();
    let exe = tmp.join("run_check");
    let out = Command::new("rustc").args(["--edition", "2021", "-C", "debug-assertions=on", "-C", "overflow-checks=on", "-o"])
        .arg(&exe).arg(&main_file).output().unwrap();
    assert!(out.status.success(), "std build failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let run = Command::new(&exe).output().unwrap();
    assert!(run.status.success(), "generated code panicked:\n{}", String::from_utf8_lossy(&run.stderr));
    // twice(3)=6.0 -> 393216 ; max+1.0 wraps ; x/0 = 0 ; x%0 = 0 ; sqrt(-1)=0 ; ln(0)=MIN ;
    // 70000 / -70000 as fixed16_16 saturate ; fixed64_64: 3*3/3 = 3, raw 3<<64 truncates to 0 in int64 -> 3+0
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "393216 -2147418113 0 0 0 -2147483648 2147483647 -2147483648 3");
}

#[test]
fn fixed_point_128_bit_math_members_are_a_transpile_error() {
    for m in ["sqrt", "sin", "cos", "tan", "exp", "ln"] {
        let src = format!("def int g(int n):\n    let a = 3 as fixed64_64\n    let b = a.{m}()\n    return 1\n");
        let (ok, err) = kernel_build_result(&format!("fixed_wide_{m}"), &src);
        assert!(!ok, "128-bit `{m}` must be rejected on --linux");
        assert!(err.contains("128-bit"), "`{m}`: unexpected diagnostic:\n{err}");
    }
}
