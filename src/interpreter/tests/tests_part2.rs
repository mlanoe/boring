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

use super::{run, run_src, get_var};
use super::*;

#[test]
fn test_enum_as_conversion() {
    let src = r#"
enum Status:
    Ok
    Err

    as string:
        match self:
            Ok: "ok"
            _: "error"

let s = Status.Ok
let _result = s as string
"#;
    assert_eq!(run_src(src), Value::Str("ok".into()));
}

#[test]
fn test_enum_pub_method() {
    let src = r#"
enum Shape:
    Circle
    Square
    Triangle

    pub def int sides():
        match self:
            Circle: 0
            Square: 4
            Triangle: 3

let sh = Shape.Triangle
let _result = sh.sides()
"#;
    assert_eq!(run_src(src), Value::Int(3));
}

#[test]
fn test_pub_field_syntax() {
    let src = r#"
struct Person:
    pub let string name
    pub var int age

var p = Person(name = "Alice", age = 30)
p.age = 31
let _name = p.name
let _age = p.age
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_name"), Value::Str("Alice".into()));
    assert_eq!(get_var(&interp, "_age"), Value::Int(31));
}

#[test]
fn test_format_float_precision() {
    let src = r#"
let pi = 3.14159
let _result = "{pi:.2}"
"#;
    assert_eq!(run_src(src), Value::Str("3.14".into()));
}

#[test]
fn test_format_float_no_decimal() {
    let src = r#"
let x = 2.9
let _result = "{x:.0}"
"#;
    assert_eq!(run_src(src), Value::Str("3".into()));
}

#[test]
fn test_format_int_zero_pad() {
    let src = r#"
let n = 42
let _result = "{n:05}"
"#;
    assert_eq!(run_src(src), Value::Str("00042".into()));
}

#[test]
fn test_format_int_width() {
    let src = r#"
let n = 42
let _result = "{n:6}"
"#;
    assert_eq!(run_src(src), Value::Str("    42".into()));
}

#[test]
fn test_format_scientific() {
    let src = r#"
let x = 12345.6789
let _result = "{x:.2e}"
"#;
    assert_eq!(run_src(src), Value::Str("1.23e4".into()));
}

#[test]
fn test_format_sign_plus() {
    let src = r#"
let x = 3.14
let _result = "{x:+.2}"
"#;
    assert_eq!(run_src(src), Value::Str("+3.14".into()));
}

#[test]
fn test_format_mixed_string() {
    let src = r#"
let price = 9.5
let qty = 3
let _result = "{qty:2} items at {price:.2} each"
"#;
    assert_eq!(run_src(src), Value::Str(" 3 items at 9.50 each".into()));
}

#[test]
fn test_space_around_eq() {
    // spaces around = in labeled args and dict literals
    let src = r#"
struct Person:
    pub let string name
    pub var int age

var p = Person(name = "Alice", age = 30)
let _name = p.name
let _age = p.age

let d = { "key" = "val" }
let _val = d["key"]
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_name"), Value::Str("Alice".into()));
    assert_eq!(get_var(&interp, "_age"), Value::Int(30));
    assert_eq!(get_var(&interp, "_val"), Value::Str("val".into()));
}

#[test]
fn test_dict_set_mutating() {
    let src = r#"
var d = { "a" = 1 }
d.set("b", 2)
d.put("c", 3)
let _a = d["a"]
let _b = d["b"]
let _c = d["c"]
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_a"), Value::Int(1));
    assert_eq!(get_var(&interp, "_b"), Value::Int(2));
    assert_eq!(get_var(&interp, "_c"), Value::Int(3));
}

#[test]
fn test_set_equality() {
    let src = r#"
let a = {1, 2, 3}
let b = {3, 1, 2}
let _eq = a == b
let _neq = a == {1, 2}
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_eq"),  Value::Bool(true));
    assert_eq!(get_var(&interp, "_neq"), Value::Bool(false));
}

#[test]
fn test_is_reference_identity() {
    // Dog b = a.clone() → b and a share the same instance
    let src = r#"
struct Dog:
    string name

let a = Dog(name = "Rex")
let b = a.clone()
let _same = b is a
let c = Dog(name = "Rex")
let _diff = c is a
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_same"), Value::Bool(true));
    assert_eq!(get_var(&interp, "_diff"), Value::Bool(false));
}

#[test]
fn test_is_type_check() {
    let src = r#"
struct Dog:
    string name

let d = Dog(name = "Rex")
let _result = d is Dog
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_result"), Value::Bool(true));
}

#[test]
fn test_is_nil() {
    let src = r#"
let x = nil
let y = 42
let _nil = x is nil
let _not_nil = y is not nil
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_nil"),     Value::Bool(true));
    assert_eq!(get_var(&interp, "_not_nil"), Value::Bool(true));
}

// ─── boring stdlib tests ──────────────────────────────────────────────────────

#[test]
fn test_use_math_namespace() {
    let src = r#"
let _result = sqrt(4.0)
"#;
    assert_eq!(run_src(src), Value::Float(2.0));
}

#[test]
fn test_use_math_direct() {
    let src = r#"
let _result = sqrt(9.0)
"#;
    assert_eq!(run_src(src), Value::Float(3.0));
}

#[test]
fn test_use_math_pi() {
    let src = r#"
let _result = PI > 3.14
"#;
    assert_eq!(run_src(src), Value::Bool(true));
}

#[test]
fn test_use_math_trig() {
    let src = r#"
let _result = round(sin(PI / 2.0) * 100.0)
"#;
    assert_eq!(run_src(src), Value::Int(100));
}

#[test]
fn test_use_result() {
    let src = r#"
def int divide(int a, int b) throws:
    if b == 0: throw "division by zero"
    a / b

let _result = try divide(10, 2) else -1
"#;
    assert_eq!(run_src(src), Value::Int(5));
}

#[test]
fn test_use_collections() {
    let src = r#"
var s = []
s.push(1)
s.push(2)
s.push(3)
let _result = s.length
"#;
    assert_eq!(run_src(src), Value::Int(3));
}

// ── Selective module imports: use a.b.c.X  /  use a.b.c.X, Y ───────────────

// Single item — `use std.collections.HashMap`
#[test]
fn test_selective_import_single() {
    let src = r#"
use std.collections.HashMap
var m = HashMap()
let _result = 42
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

// Multiple std items.
#[test]
fn test_selective_import_result() {
    let src = r#"
use std.collections.HashMap, HashSet
var m = HashMap()
var s = HashSet()
let _result = 99
"#;
    assert_eq!(run_src(src), Value::Int(99));
}

// Multiple items from std.collections.
#[test]
fn test_selective_import_multiple() {
    let src = r#"
use std.collections.HashMap, HashSet
var m = HashMap()
var s = HashSet()
let _result = 2
"#;
    assert_eq!(run_src(src), Value::Int(2));
}

// sqrt is a global builtin — no import needed.
#[test]
fn test_selective_import_dot_form() {
    let src = r#"
let _result = sqrt(16.0)
"#;
    assert_eq!(run_src(src), Value::Float(4.0));
}

// ─── req / def mutability tests ──────────────────────────────────────────────

#[test]
fn test_req_callable_on_let() {
    // `let` binding can call `req` methods
    let src = r#"
struct Counter:
    int value

    req int doubled(): self.value * 2

let c = Counter(value= 5)
let _result = c.doubled()
"#;
    assert_eq!(run_src(src), Value::Int(10));
}

#[test]
fn test_def_requires_var() {
    // calling `def` on a `let` binding must produce a runtime error
    let src = r#"
struct Counter:
    var int value

    def void increment():
        self.value = self.value + 1

let c = Counter(value= 0)
c.increment()
"#;
    let (_interp, res) = run(src);
    assert!(res.is_err(), "calling def on let binding should error");
    let msg = res.unwrap_err().message;
    assert!(msg.contains("cannot call mutating method"), "unexpected error: {}", msg);
}

#[test]
fn test_req_as_property() {
    // zero-param `req` callable without `()` (property-style access)
    let src = r#"
struct Circle:
    float radius

    req float area(): 3.14159 * self.radius * self.radius

let c = Circle(radius= 1.0)
let _result = c.area
"#;
    // area = pi * 1 * 1 ≈ 3.14159
    let val = run_src(src);
    if let Value::Float(f) = val {
        assert!((f - 3.14159).abs() < 0.001, "expected ~3.14159, got {}", f);
    } else {
        panic!("expected Float, got {:?}", val);
    }
}

#[test]
fn test_transient_in_req() {
    // transient field writable inside a req method (lazy/cache pattern)
    let src = r#"
struct Cache:
    int base
    transient int? _cached = nil

    req int value():
        if let v = self._cached:
            return v
        self._cached = self.base * 2
        self._cached else 0

let c = Cache(base= 21)
let _result = c.value()
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

// ─── init constructor tests ───────────────────────────────────────────────────

#[test]
fn test_init_form2_pub_fields() {
    // Form 2: init with pub field-declaring params (no body)
    let src = r#"
struct Person:
    init(pub String name, pub Int age = 0)

let p = Person("Alice", 30)
let _result = p.name
"#;
    assert_eq!(run_src(src), Value::Str("Alice".to_string()));
}

#[test]
fn test_init_form2_default_value() {
    // Form 2: default value on pub param
    let src = r#"
struct Person:
    init(pub String name, pub Int age = 0)

let p = Person("Bob")
let _result = p.age
"#;
    assert_eq!(run_src(src), Value::Int(0));
}

#[test]
fn test_init_form3_with_body() {
    // Form 3: init with body, fields declared separately
    let src = r#"
struct Circle:
    pub float radius
    pub float area
    init(float r):
        self.radius = r
        self.area = 3.14159 * r * r

let c = Circle(1.0)
let _result = c.radius
"#;
    assert_eq!(run_src(src), Value::Float(1.0));
}

#[test]
fn test_init_form3_computed_field() {
    // Form 3: init body computes a field
    let src = r#"
struct Circle:
    pub float radius
    pub float area
    init(float r):
        self.radius = r
        self.area = 3.14159 * r * r

let c = Circle(2.0)
let _result = c.area
"#;
    let val = run_src(src);
    if let Value::Float(f) = val {
        assert!((f - 3.14159 * 4.0).abs() < 0.001, "expected ~12.566, got {}", f);
    } else {
        panic!("expected Float, got {:?}", val);
    }
}

#[test]
fn test_init_private_field() {
    // No-body init: param without `pub` → private field (accessible by self, not exported)
    let src = r#"
struct Secret:
    init(string value)

    req string reveal(): self.value

let s = Secret("hidden")
let _result = s.reveal()
"#;
    assert_eq!(run_src(src), Value::Str("hidden".to_string()));
}

#[test]
fn test_init_private_mutable_field() {
    // No-body init: `var` without `pub` → private mutable field
    let src = r#"
struct Counter:
    init(var int count = 0)

    def void increment(): self.count = self.count + 1
    req int value(): self.count

var c = Counter()
c.increment()
c.increment()
let _result = c.value()
"#;
    assert_eq!(run_src(src), Value::Int(2));
}

// ─── Nested function tests ────────────────────────────────────────────────────

#[test]
fn test_nested_fn_basic() {
    // A function declared inside another is usable locally
    let src = r#"
def int compute(int x):
    def int double(int n): n * 2
    def int addOne(int n): n + 1
    double(addOne(x))

let _result = compute(20)
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_nested_fn_captures_outer() {
    // A nested function closes over the outer function's locals
    let src = r#"
def int makeAdder(int base):
    def int add(int n): base + n
    add(10)

let _result = makeAdder(32)
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_nested_fn_invisible_outside() {
    // A nested function is not visible outside its enclosing function
    let src = r#"
def int outer():
    def int inner(): 1
    inner()

let _result = outer()
"#;
    // outer() works fine
    assert_eq!(run_src(src), Value::Int(1));

    // inner is not accessible at top level
    let src2 = r#"
def int outer():
    def int inner(): 1
    inner()

let _r = inner()
"#;
    let tokens = crate::lexer::lex(src2).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    let mut interp = Interpreter::new();
    let result = interp.exec_program(&program);
    assert!(result.is_err(), "inner() should not be visible outside outer()");
}

// ─── Nested struct / enum tests ───────────────────────────────────────────────

#[test]
fn test_nested_struct_basic() {
    // A struct declared inside a function is usable locally
    let src = r#"
def int makePoint():
    struct Point:
        int x
        int y
    let p = Point(x = 3, y = 4)
    p.x + p.y

let _result = makePoint()
"#;
    assert_eq!(run_src(src), Value::Int(7));
}

#[test]
fn test_nested_struct_invisible_outside() {
    // A locally declared struct is not visible outside
    let src = r#"
def int outer():
    struct Local:
        int v
    let l = Local(v= 1)
    l.v

let _r = outer()
let _bad = Local(v= 2)
"#;
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    let mut interp = Interpreter::new();
    let result = interp.exec_program(&program);
    assert!(result.is_err(), "Local should not be visible outside outer()");
}

#[test]
fn test_nested_enum_basic() {
    // An enum declared inside a function is usable locally
    let src = r#"
def string describeColor():
    enum Color:
        Red
        Green
        Blue
    let c = Color.Red
    match c:
        Red: "red"
        _: "other"

let _result = describeColor()
"#;
    assert_eq!(run_src(src), Value::Str("red".to_string()));
}

// ─── Builtins: panic / assert_eq / assert_neq ────────────────────────────────

#[test]
fn test_panic_triggers_error() {
    let src = r#"panic("something went wrong")"#;
    let (_, res) = run(src);
    assert!(res.is_err());
    assert!(res.unwrap_err().message.contains("something went wrong"));
}

#[test]
fn test_assert_eq_pass() {
    let src = r#"assert_eq(1 + 1, 2)"#;
    let (_, res) = run(src);
    res.expect("assert_eq should pass");
}

#[test]
fn test_assert_eq_fail() {
    let src = r#"assert_eq(1, 2)"#;
    let (_, res) = run(src);
    let err = res.unwrap_err();
    assert!(err.message.contains("1") || err.message.contains("2"));
}

#[test]
fn test_assert_eq_custom_message() {
    let src = r#"assert_eq(1, 2, "values differ")"#;
    let (_, res) = run(src);
    assert!(res.is_err());
    assert!(res.unwrap_err().message.contains("values differ"));
}

#[test]
fn test_assert_neq_pass() {
    let src = r#"assert_neq(1, 2)"#;
    let (_, res) = run(src);
    res.expect("assert_neq should pass");
}

#[test]
fn test_assert_neq_fail() {
    let src = r#"assert_neq(42, 42)"#;
    let (_, res) = run(src);
    assert!(res.is_err());
    assert!(res.unwrap_err().message.contains("42"));
}

// ─── main entry point tests ───────────────────────────────────────────────────

#[test]
fn test_main_basic() {
    // `main` is called automatically after top-level declarations
    let src = r#"
var result = 0

def void main():
    result = 42
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(42));
}

#[test]
fn test_main_throws() {
    // `def main() throws:` — uncaught exception becomes a runtime error
    let src = r#"
def void main() throws:
    throw "oops"
"#;
    let (_, res) = run(src);
    assert!(res.is_err());
    assert!(res.unwrap_err().message.contains("oops"));
}

#[test]
fn test_main_task() {
    // `task def main():` — may call other task functions
    let src = r#"
var result = 0

task def int compute(): 42

task def void main():
    result = compute()
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(42));
}

#[test]
fn test_main_task_throws() {
    // `def main() task throws:` — both modifiers together
    let src = r#"
var result = 0

task def int work() throws:
    42

task def void main() throws:
    result = try work() else 0
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(42));
}

#[test]
fn test_no_main_scripting_mode() {
    // Without main, top-level statements execute directly (scripting mode)
    let src = r#"
let result = 21 * 2
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(42));
}

// ─── Local use alias ─────────────────────────────────────────────────────────

#[test]
fn test_local_use_alias_in_function() {
    // `use` alias at top level simplifies fn-type parameters; `task` is a prefix qualifier
    let src = r#"
use Worker as task int () throws

task def int run(Worker f):
    try f() else 0

task def int work() throws: 42

var result = run(work)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(42));
}

#[test]
fn test_local_use_alias_struct_type() {
    // `use` alias inside a function, used to type a nested function parameter
    let src = r#"
struct Point:
    pub int x
    pub int y

def void main():
    use Pt as Point'
    def int describe(Pt p): p.x + p.y
    let p = Point(x = 10, y = 32)
    assert_eq(describe(p), 42)
"#;
    let (_, res) = run(src);
    res.expect("no runtime error");
}

#[test]
fn test_top_level_fn_type_alias() {
    // Top-level `use` alias for a function type used as parameter
    let src = r#"
use Transformer as int (int) throws

def int apply(int x, Transformer f): try f(x) else 0

def int double(int x) throws: x * 2

var result = apply(21, double)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(42));
}

// ─── task shorthand (task RetType name instead of task def RetType name) ──────

#[test]
fn test_task_shorthand_top_level() {
    // `task int compute()` is identical to `task def int compute()`
    let src = r#"
task int double(int x): x * 2
var result = double(21)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(42));
}

#[test]
fn test_task_shorthand_void() {
    // `task void` — shorthand compiles and runs without error
    let src = r#"
task void noop(int x):
    let _ = x
noop(42)
let _result = 1
"#;
    assert_eq!(run_src(src), Value::Int(1));
}

#[test]
fn test_task_shorthand_struct_method() {
    // `task int method()` inside a struct body
    let src = r#"
struct Counter:
    pub int value = 0
    task int inc(): self.value + 1

let c = Counter()
let _result = c.inc()
"#;
    assert_eq!(run_src(src), Value::Int(1));
}

// ─── req / def function types ────────────────────────────────────────────────

#[test]
fn test_fn_type_req_accepted_as_def() {
    // A pure `req` function can be passed where a `def` (mutating) is expected — subtyping.
    let src = r#"
def int apply(def int(int) f, int x): f(x)
def int double(int n) throws: n * 2
let _result = apply(double, 21)
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_fn_type_req_alias() {
    // `req` in a type alias
    let src = r#"
use Pure as req int (int)
def int apply(Pure f, int x): f(x)
def int triple(int n) throws: n * 3
let _result = apply(triple, 14)
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_fn_type_def_explicit() {
    // Explicit `def` prefix is the same as no prefix
    let src = r#"
use Mutating as def int (int)
def int apply(Mutating f, int x): f(x)
def int inc(int n) throws: n + 1
let _result = apply(inc, 41)
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_fn_type_req_task_combined() {
    // `req task` — pure async function type
    let src = r#"
use PureWorker as req task int () throws
task def int run(PureWorker f): try f() else 0
task def int work() throws: 42
var result = run(work)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(42));
}

// ─── Closure task/throws modifiers ───────────────────────────────────────────

#[test]
fn test_closure_explicit_task() {
    // `() task: body` — explicit task modifier accepted on closure
    let src = r#"
def int run_task(int f() task): f()

var result = run_task(() task: 42)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(42));
}

#[test]
fn test_closure_explicit_throws() {
    // `() throws: body` — explicit throws modifier accepted on closure
    let src = r#"
def int safe(int f() throws): try f() else 0

var result = safe(() throws: 42)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(42));
}

#[test]
fn test_closure_task_throws_both() {
    // `() task throws: body` — both modifiers together
    let src = r#"
def int run(int f() task throws): try f() else 0

var result = run(() task throws: 42)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(42));
}

// ─── Multi-line closure declarations ─────────────────────────────────────────

#[test]
fn test_closure_multiline_untyped_params() {
    // Untyped params spread across multiple lines
    let src = r#"
let add = (
    x,
    y
): x + y

let _result = add(3, 4)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(7));
}

#[test]
fn test_closure_multiline_typed_params() {
    // Typed params spread across multiple lines
    let src = r#"
let mul = (
    int x,
    int y
): x * y

let _result = mul(5, 6)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(30));
}

#[test]
fn test_closure_multiline_call_args() {
    // Call with multi-line argument list
    let src = r#"
def int add(int x, int y): x + y

let _result = add(
    10,
    20
)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(30));
}

#[test]
fn test_closure_multiline_passed_as_arg() {
    // Multi-line closure passed as argument
    let src = r#"
def int apply(int x, int f(int)): f(x)

let _result = apply(
    7,
    (
        int n
    ): n * n
)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(49));
}

// ─── Enum trait conformance ───────────────────────────────────────────────────

#[test]
fn test_enum_protocol_header() {
    // `enum Color as Drawable:` — conformance declared in header
    let src = r#"
trait Drawable:
    req string draw()

enum Color as Drawable:
    Red
    Blue

    req string draw():
        "painted"

let c = Color.Red
let result = c.draw()
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Str("painted".to_string()));
}

#[test]
fn test_enum_conformance_block() {
    // conformance via ext block
    let src = r#"
trait Drawable:
    req string draw()

enum Shape:
    Circle
    Square

ext Shape as Drawable:
    req string draw():
        "shape drawn"

let s = Shape.Circle
let result = s.draw()
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Str("shape drawn".to_string()));
}

#[test]
fn test_enum_multiple_protocols() {
    // `enum Dir as Printable, Hashable:` — multiple trait conformances
    let src = r#"
trait Printable:
    req string describe()

enum Dir as Printable:
    North
    South

    req string describe():
        "a direction"

let d = Dir.North
let result = d.describe()
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Str("a direction".to_string()));
}

#[test]
fn test_enum_structural_conformance() {
    // Enum with matching method satisfies trait structurally (no explicit `as`)
    let src = r#"
trait Drawable:
    req string draw()

enum Color:
    Red
    Blue

    req string draw(): "color drawn"

def string render<T as Drawable>(T item):
    item.draw()

let result = render(Color.Red)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Str("color drawn".to_string()));
}

// ─── Move semantics: `let b' = a` ────────────────────────────────────────────

#[test]
fn test_let_borrow_default_keeps_both() {
    // `let b = a` (no tick) — borrow by default: both variables remain valid
    let src = r#"
let a = 42
let b = a
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "a"), Value::Int(42)); // still valid
    assert_eq!(get_var(&interp, "b"), Value::Int(42));
}

#[test]
fn test_let_move_transfers_value() {
    // `let b = a` on a struct — implicit move — b gets the object
    let src = r#"
struct Box:
    init(pub int val)

let a = Box(val = 42)
let b = a
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert!(matches!(get_var(&interp, "b"), Value::Object(_)));
}

#[test]
fn test_let_move_invalidates_source() {
    // `let b = a` on a struct — source `a` is invalidated after the move
    let src = r#"
struct Box:
    init(pub int val)

let a = Box(val = 42)
let b = a
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert!(matches!(get_var(&interp, "a"), Value::Moved(_)));
}

#[test]
fn test_let_move_source_access_fails() {
    // Accessing a moved struct variable must produce a runtime error
    let src = r#"
struct Box:
    init(pub int val)

def int bad():
    let a = Box(val = 42)
    let b = a
    return a.val

let result = bad()
"#;
    let (_interp, res) = run(src);
    assert!(res.is_err(), "accessing a moved variable should fail at runtime");
}

#[test]
fn test_let_move_struct() {
    // Move works with struct values: source is invalidated, destination has the object
    let src = r#"
struct Dog:
    init(pub string name)

let a = Dog(name = "Rex")
let b = a
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert!(matches!(get_var(&interp, "a"), Value::Moved(_)));
    assert!(matches!(get_var(&interp, "b"), Value::Object(_)));
}

#[test]
fn test_var_move_mutable() {
    // `var b = a` — mutable move: source invalidated, binding is mutable
    let src = r#"
struct Point:
    init(pub var int x, pub var int y)

let a = Point(x = 1, y = 2)
var b = a
b.x = 10
let result = b.x
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(10));
    assert!(matches!(get_var(&interp, "a"), Value::Moved(_)));
}

#[test]
fn test_let_move_inside_fn() {
    // Move of an owned struct inside a function body
    let src = r#"
struct Token:
    init(pub int id)

def int transfer():
    let a = Token(id = 7)
    let b = a
    return b.id

let result = transfer()
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(7));
}

#[test]
fn test_let_move_copy_type_noop() {
    // `let b = a` on a copy type (int): a is NOT invalidated — copy types are never moved
    let src = r#"
let a = 99
let b = a
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "b"), Value::Int(99));
    assert_eq!(get_var(&interp, "a"), Value::Int(99)); // still valid — copy, not moved
}

#[test]
fn test_let_clone_keeps_source_accessible() {
    // `.clone()` performs an explicit deep copy — source remains accessible
    let src = r#"
struct Box:
    init(pub int val)

let a = Box(val = 42)
let b = a.clone()
let result = a.val
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(42));
    assert!(matches!(get_var(&interp, "b"), Value::Object(_)));
}

// ─── T'shared'weak / T'actor'weak — compound weak references ──────

#[test]
fn test_weak_type_in_signature() {
    // `Dog'shared'weak` in function return type and parameter type annotations
    let src = r#"
struct Dog:
    init(pub string name)

def Dog'shared'weak get_weak(Dog'shared d):
    let Dog'shared'weak w = d
    w

let Dog'shared d = Dog(name = "Rex")
let w = get_weak(d)
"#;
    let (_interp, res) = run(src);
    res.expect("no runtime error");
}

#[test]
fn test_weak_upgrade_returns_object() {
    // `.upgrade()` on a compound weak ref returns the object (live in interpreter)
    let src = r#"
struct Node:
    init(pub int val)

let Node'shared         strong  = Node(val = 42)
let Node'shared'weak    weak    = strong
let upgraded                    = weak.upgrade()
let result                      = upgraded.val
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(42));
}

#[test]
fn test_weak_not_task_safe() {
    // T'shared'weak is not task-safe (Weak<T> — non-owning)
    use crate::ast::{Type, OwnerQual};
    let ty = Type::Qualified(
        Box::new(Type::Qualified(Box::new(Type::Named("Dog".into())), OwnerQual::Shared)),
        OwnerQual::Weak,
    );
    assert!(!ty.is_task_safe());
}

#[test]
fn test_weak_shared_compound() {
    // `'weak` infers qualifier from RHS — `a` is `'shared` so `b` becomes `'shared'weak`
    let src = r#"
struct Dog:
    init(pub string name)

let a'shared = Dog(name = "Rex")
let b'weak = a
let upgraded = b.upgrade()
let _result = upgraded.name
"#;
    assert_eq!(run_src(src), Value::Str("Rex".into()));
}

#[test]
fn test_weak_explicit_compound_type() {
    // Explicit `Dog'shared'weak` in type annotation
    let src = r#"
struct Dog:
    init(pub string name)

let Dog'shared strong = Dog(name = "Buddy")
let Dog'shared'weak  w = strong
let upgraded = w.upgrade()
let _result = upgraded.name
"#;
    assert_eq!(run_src(src), Value::Str("Buddy".into()));
}

// ─── var in function parameters ──────────────────────────────────────────────

#[test]
fn test_var_param_reassignable() {
    // `var` on a param makes it reassignable inside the function body
    let src = r#"
def int clamp(var int x, int lo, int hi):
    if x < lo: x = lo
    if x > hi: x = hi
    return x

let r1 = clamp(5, 0, 10)
let r2 = clamp(-3, 0, 10)
let r3 = clamp(42, 0, 10)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "r1"), Value::Int(5));
    assert_eq!(get_var(&interp, "r2"), Value::Int(0));
    assert_eq!(get_var(&interp, "r3"), Value::Int(10));
}

#[test]
fn test_immutable_param_reassign_fails() {
    // Without `var`, reassigning a param must fail
    let src = r#"
def int bad(int x):
    x = 99
    return x

let result = bad(1)
"#;
    let (_interp, res) = run(src);
    assert!(res.is_err(), "reassigning an immutable param should fail");
}

// ─── Multi-clause if let / guard let ─────────────────────────────────────────

#[test]
fn test_if_let_single_clause() {
    // Existing single-clause if let still works
    let src = r#"
def string greet(string? name):
    if let n = name: return "hello " + n
    return "nobody"

let r1 = greet("Alice")
let r2 = greet(nil)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "r1"), Value::Str("hello Alice".into()));
    assert_eq!(get_var(&interp, "r2"), Value::Str("nobody".into()));
}

#[test]
fn test_if_let_multi_binding() {
    // `if let x = a, let y = b:` — both bindings must succeed
    let src = r#"
def string both(string? a, string? b):
    if let x = a, let y = b:
        return x + " " + y
    return "missing"

let r1 = both("hello", "world")
let r2 = both("hello", nil)
let r3 = both(nil, "world")
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "r1"), Value::Str("hello world".into()));
    assert_eq!(get_var(&interp, "r2"), Value::Str("missing".into()));
    assert_eq!(get_var(&interp, "r3"), Value::Str("missing".into()));
}

#[test]
fn test_if_let_with_bool_condition() {
    // `if let x = a, x > 0:` — binding + boolean condition
    let src = r#"
def string check(int? val):
    if let x = val, x > 0:
        return "positive"
    return "nope"

let r1 = check(5)
let r2 = check(-3)
let r3 = check(nil)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "r1"), Value::Str("positive".into()));
    assert_eq!(get_var(&interp, "r2"), Value::Str("nope".into()));
    assert_eq!(get_var(&interp, "r3"), Value::Str("nope".into()));
}

#[test]
fn test_if_let_second_uses_first() {
    // Bindings are sequential: `y` can reference `x`
    let src = r#"
def int sum(int? a, int? b):
    if let x = a, let y = b:
        return x + y
    return 0

let result = sum(10, 32)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "result"), Value::Int(42));
}

#[test]
fn test_guard_let_multi_binding() {
    // `guard let x = a, let y = b else:` — both bindings visible after guard
    let src = r#"
def string concat(string? a, string? b):
    guard let x = a, let y = b else: return "missing"
    return x + y

let r1 = concat("foo", "bar")
let r2 = concat("foo", nil)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "r1"), Value::Str("foobar".into()));
    assert_eq!(get_var(&interp, "r2"), Value::Str("missing".into()));
}

#[test]
fn test_guard_let_with_bool_condition() {
    // `guard let x = a, x > 0 else:` — binding + boolean guard
    let src = r#"
def int safe_sqrt(int? val):
    guard let x = val, x >= 0 else: return -1
    return x

let r1 = safe_sqrt(9)
let r2 = safe_sqrt(-1)
let r3 = safe_sqrt(nil)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "r1"), Value::Int(9));
    assert_eq!(get_var(&interp, "r2"), Value::Int(-1));
    assert_eq!(get_var(&interp, "r3"), Value::Int(-1));
}

// ─── for without binding variable ────────────────────────────────────────────

#[test]
fn test_for_range_no_var() {
    // `for 1..4:` — exclusive range without variable (1, 2, 3)
    let src = r#"
var count = 0
for 1..4:
    count = count + 1
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "count"), Value::Int(3));
}

#[test]
fn test_for_range_inclusive_no_var() {
    // `for 1..=4:` — inclusive range without variable (1, 2, 3, 4)
    let src = r#"
var count = 0
for 1..=4:
    count = count + 1
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "count"), Value::Int(4));
}

#[test]
fn test_for_with_var_still_works() {
    // `for i in 1..4:` — exclusive range with variable
    let src = r#"
var sum = 0
for i in 1..4:
    sum = sum + i
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "sum"), Value::Int(6));
}

// ─── pass / native ────────────────────────────────────────────────────────────

#[test]
fn test_pass_fn() {
    // `def foo(): pass` — empty body, calling it returns nil
    let _src = r#"
def void foo():
    pass
let _result = foo()
"#;
    // `pass` used as a stmt inside a block body is just parsed as an ident expression today.
    // The inline form `def foo(): pass` produces an empty body.
    let src2 = r#"
def void foo(): pass
let _result = foo()
"#;
    let val = run_src(src2);
    assert_eq!(val, Value::Void);
}

#[test]
fn test_pass_fn_inline() {
    // Inline `pass` — void function returns Void
    let src = r#"
def void noop(int x): pass
let _result = noop(42)
"#;
    let val = run_src(src);
    assert_eq!(val, Value::Void);
}

#[test]
fn test_pass_struct() {
    // `struct Empty: pass` — can be declared without error
    let src = r#"
struct Empty: pass
let e = Empty()
let _result = 1
"#;
    let val = run_src(src);
    assert_eq!(val, Value::Int(1));
}

#[test]
fn test_pass_enum() {
    // `enum Void: pass` — empty enum, no error
    let src = r#"
enum Nothing: pass
let _result = 2
"#;
    let val = run_src(src);
    assert_eq!(val, Value::Int(2));
}

#[test]
fn test_native_fn_does_not_shadow_builtin() {
    // Declaring `print` as native must not overwrite the real built-in.
    let src = r#"
def void print(string... texts): native
let _result = 3
"#;
    let val = run_src(src);
    assert_eq!(val, Value::Int(3));
    // Ensure `print` is still the built-in NativeFn, not a user Fn.
    let tokens = crate::lexer::lex(src).unwrap();
    let program = crate::parser::parse(tokens).unwrap();
    let mut interp = Interpreter::new();
    interp.exec_program(&program).unwrap();
    let v = interp.global.borrow().get("print").unwrap();
    assert!(matches!(v, Value::NativeFn { .. }), "print must remain a NativeFn");
}

#[test]
fn test_native_fn_new() {
    // A brand-new `native` function is simply not registered (undefined).
    let src = r#"
def int myRuntimeFn(int x): native
let _result = 5
"#;
    let val = run_src(src);
    assert_eq!(val, Value::Int(5));
    // Ensure `myRuntimeFn` is NOT defined (native stub = not in env)
    let tokens2 = crate::lexer::lex(src).unwrap();
    let program2 = crate::parser::parse(tokens2).unwrap();
    let mut interp2 = Interpreter::new();
    interp2.exec_program(&program2).unwrap();
    let v = interp2.global.borrow().get("myRuntimeFn");
    assert!(v.is_none(), "native stubs must not be registered in the env");
}

// ─── match guards ─────────────────────────────────────────────────────────────

#[test]
fn test_match_guard_basic() {
    // Guard on a literal pattern
    let src = r#"
def string classify(int n):
    match n:
        0: "zero"
        x if x > 0: "positive"
        _: "negative"

let _result = classify(5)
"#;
    assert_eq!(run_src(src), Value::Str("positive".into()));
}

#[test]
fn test_match_guard_negative() {
    let src = r#"
def string classify(int n):
    match n:
        0: "zero"
        x if x > 0: "positive"
        _: "negative"

let _result = classify(-3)
"#;
    assert_eq!(run_src(src), Value::Str("negative".into()));
}

#[test]
fn test_match_guard_zero() {
    let src = r#"
def string classify(int n):
    match n:
        0: "zero"
        x if x > 0: "positive"
        _: "negative"

let _result = classify(0)
"#;
    assert_eq!(run_src(src), Value::Str("zero".into()));
}

#[test]
fn test_match_guard_with_enum() {
    // Guard on an enum variant binding
    let src = r#"
enum Opt:
    Some(int value)
    None

def string describe(Opt'shared o):
    match o:
        Some(n) if n > 10: "big"
        Some(n): "small"
        None: "nothing"

let _result = describe(Opt.Some(42))
"#;
    assert_eq!(run_src(src), Value::Str("big".into()));
}

#[test]
fn test_match_guard_enum_small() {
    let src = r#"
enum Opt:
    Some(int value)
    None

def string describe(Opt'shared o):
    match o:
        Some(n) if n > 10: "big"
        Some(n): "small"
        None: "nothing"

let _result = describe(Opt.Some(3))
"#;
    assert_eq!(run_src(src), Value::Str("small".into()));
}

#[test]
fn test_match_guard_or_patterns() {
    // Guard applies to all OR-alternatives in one arm
    let src = r#"
def string check(int n):
    match n:
        1 | 2 | 3 if true: "one-two-three"
        _: "other"

let _result = check(2)
"#;
    assert_eq!(run_src(src), Value::Str("one-two-three".into()));
}

#[test]
fn test_match_guard_fallthrough() {
    // Guard fails → falls through to next arm
    let src = r#"
def string check(int n):
    match n:
        x if x > 100: "huge"
        x if x > 10: "big"
        _: "small"

let _result = check(50)
"#;
    assert_eq!(run_src(src), Value::Str("big".into()));
}

#[test]
fn test_native_struct() {
    // `struct NativeStr: native` — no error, struct is declared as a native stub
    let src = r#"
struct NativeStr: native
let _result = 7
"#;
    let val = run_src(src);
    assert_eq!(val, Value::Int(7));
}

// ─── Lifetime annotations ─────────────────────────────────────────────────────

#[test]
fn test_lifetime_on_field() {
    // `struct Parser<&a>: \n  string&a source`
    // Lifetime is purely an annotation; the interpreter treats it as a plain string.
    let src = r#"
struct Parser<&a>:
    string&a source

let p = Parser("hello")
let _result = p.source
"#;
    assert_eq!(run_src(src), Value::Str("hello".into()));
}

#[test]
fn test_lifetime_on_fn_param() {
    // `<&a>` is optional — lifetimes are inferred from the signature automatically.
    let src = r#"
def string&a longest(string&a x, string&a y):
    if x.len() > y.len(): return x
    return y

let _result = longest("hi", "hello")
"#;
    assert_eq!(run_src(src), Value::Str("hello".into()));
}

#[test]
fn test_lifetime_explicit_decl_still_works() {
    // `<&a>` explicit declaration is still valid (optional but supported).
    let src = r#"
def string&a first<&a>(string&a x, string&a y):
    return x

let _result = first("hello", "world")
"#;
    assert_eq!(run_src(src), Value::Str("hello".into()));
}

#[test]
fn test_owned_single_letter_param_not_lifetime() {
    // `Dog' d` — bare tick (owned) + param named `d` must NOT consume `d` as a lifetime.
    let src = r#"
struct Dog:
    string name

def string pet(Dog' d):
    d.name

let d = Dog("Rex")
let _result = pet(d)
"#;
    assert_eq!(run_src(src), Value::Str("Rex".into()));
}

#[test]
fn test_lifetime_mixed_with_type_param() {
    // `struct Wrapper<T, &a>:` — type param + lifetime param together
    let src = r#"
struct Wrapper<T, &a>:
    T&a value

let w = Wrapper(42)
let _result = w.value
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_lifetime_multiple() {
    // Multiple lifetime params: `<&a, &b>`
    let src = r#"
struct Pair<&a, &b>:
    string&a first
    string&b second

let p = Pair("foo", "bar")
let _result = p.first
"#;
    assert_eq!(run_src(src), Value::Str("foo".into()));
}

// ─── throws with error type ───────────────────────────────────────────────────

#[test]
fn test_throws_typed_parses() {
    // `throws MyError` — parses without error; runtime behaves identically to `throws`.
    let src = r#"
struct MyError:
    string message

def int divide(int a, int b) throws MyError:
    if b == 0: throw MyError("division by zero")
    return a / b

let _result = try divide(10, 2) else -1
"#;
    assert_eq!(run_src(src), Value::Int(5));
}

#[test]
fn test_throws_typed_propagates() {
    // Thrown value propagates through `try/catch` as before.
    let src = r#"
struct IOError:
    string message

def string readFile(string path) throws IOError:
    throw IOError("not found")

var _msg = "ok"
try:
    readFile("x.txt")
catch:
    _msg = "caught"

let _result = _msg
"#;
    assert_eq!(run_src(src), Value::Str("caught".into()));
}

#[test]
fn test_throws_typed_and_task() {
    // Canonical order: `task` before `throws ErrorType`.
    let src = r#"
struct Err:
    string message

task def int foo() throws Err:
    return 42

task def int bar() throws Err:
    return 99

let _result = 1
"#;
    assert_eq!(run_src(src), Value::Int(1));
}

#[test]
fn test_throws_untyped_still_works() {
    // `throws` without a type — unchanged behaviour.
    let src = r#"
def int safe(int x) throws:
    if x < 0: throw "negative"
    return x

let _result = try safe(7) else -1
"#;
    assert_eq!(run_src(src), Value::Int(7));
}

// ─── T'stack qualifier ───────────────────────────────────────────────────────

#[test]
fn test_stack_qualifier_field() {
    // 'stack on a field: transparent at runtime, meaningful for the transpiler
    let src = r#"
struct Inner:
    int val

struct Outer:
    Inner'stack inner

let o = Outer(Inner(42))
let _result = o.inner.val
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_stack_qualifier_param() {
    // 'stack on a parameter: accepted by the type checker, transparent at runtime
    let src = r#"
struct Point:
    int x
    int y

def int sum(Point'stack p):
    return p.x + p.y

let _result = sum(Point(10, 32))
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

