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

use super::*;
use crate::lexer::lex;
use crate::parser::parse;

pub(super) fn run(src: &str) -> (Interpreter, Result<(), RuntimeError>) {
    let tokens = lex(src).expect("lex error");
    let program = parse(tokens).expect("parse error");
    let mut interp = Interpreter::new();
    let result = interp.exec_program(&program);
    (interp, result)
}

pub(super) fn get_var(interp: &Interpreter, name: &str) -> Value {
    interp.global.borrow().get(name).unwrap_or(Value::Nil)
}

pub(super) fn run_src(src: &str) -> Value {
    let tokens = lex(src).expect("lex error");
    let program = parse(tokens).expect("parse error");
    let mut interp = Interpreter::new();
    interp.exec_program(&program).expect("runtime error");
    let val = interp.global.borrow().get("_result").unwrap_or(Value::Nil);
    val
}

#[test]
fn test_guard_expr() {
    let src = "
def int check(int x):
    guard x > 0 else: return -1
    return x * 2

let _result = check(5)
";
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(10));
}

#[test]
fn test_guard_expr_false() {
    let src = "
def int check(int x):
    guard x > 0 else: return -1
    return x * 2

let _result = check(-3)
";
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(-1));
}

#[test]
fn test_guard_let() {
    let src = "
def string greet(string? name):
    guard let n = name else: return \"nobody\"
    return \"hello \" + n

let _result = greet(\"Alice\")
let _nil_result = greet(nil)
";
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Str("hello Alice".to_string()));
    assert_eq!(get_var(&interp, "_nil_result"), Value::Str("nobody".to_string()));
}

#[test]
fn test_try_else_inline() {
    let src = "
def void fail() throws:
    throw 42

let _result = try fail() else 0
";
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(0));
}

#[test]
fn test_try_else_block() {
    let src = "
def void fail() throws:
    throw 42

let _result = try fail() else:
    99
";
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(99));
}

#[test]
fn test_else_nil_coalescing() {
    let src = "
let int? x = nil
let _result = x else 42
";
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(42));
}

#[test]
fn test_if_inline_no_colon_else() {
    let src = "
let x = 5
let _result = if x > 3: \"big\" else \"small\"
";
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Str("big".to_string()));
}

#[test]
fn test_task_fire_and_forget() {
    // task as statement — just runs
    let src = "
task def void work():
    let x = 42

task work()
let _result = true
";
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Bool(true));
}

#[test]
fn test_task_future_value() {
    let src = "
task def int compute():
    42

let f = task compute()
let _result = f.value
";
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(42));
}

#[test]
fn test_task_block_future() {
    let src = "
task def int double(int x):
    x * 2

let f = task:
    double(21)

let _result = f.value
";
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(42));
}

#[test]
fn test_task_colon_optional() {
    let src = "
task def int get():
    99

let a = task get()
let b = task: get()
let _result = a.value + b.value
";
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(198));
}

#[test]
fn test_no_return_in_closure() {
    // parse error expected
    let src = "
let f = (x): return x * 2
";
    let tokens = crate::lexer::lex(src).expect("lex");
    let result = crate::parser::parse(tokens);
    assert!(result.is_err(), "return in closure should be a parse error");
}

#[test]
fn test_fn_type_param_call() {
    let src = "
def int apply(int f(int), int x): f(x)
let double = (n): n * 2
let _result = apply(double, 21)
";
    if let Value::Int(n) = run_src(src) {
        assert_eq!(n, 42);
    } else {
        panic!("expected Int(42)");
    }
}

#[test]
fn test_fn_inferred() {
    // `def add(a, b):` — omitting both param types and return type is now a parse error.
    // Only closures may omit types.
    let src = "def add(a, b): a + b\nlet _result = add(3, 4)";
    let tokens = crate::lexer::lex(src).expect("lex");
    let result = crate::parser::parse(tokens);
    assert!(result.is_err(), "def without types should be a parse error");
}

#[test]
fn test_fn_typed() {
    let src = "
def int add(int a, int b): a + b
let _result = add(3, 4)
";
    assert_eq!(run_src(src), Value::Int(7));
}

#[test]
fn test_closure_inferred() {
    let src = "
let double = (x): x * 2
let _result = double(21)
";
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_closure_typed() {
    let src = "
let double = (int x): x * 2
let _result = double(21)
";
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_trailing_closure_inline() {
    let src = "
let _result = [1, 2, 3].map (x): x * 2
";
    if let Value::Array(v) = run_src(src) {
        assert_eq!(v, vec![Value::Int(2), Value::Int(4), Value::Int(6)]);
    } else { panic!("expected Array"); }
}

#[test]
fn test_trailing_closure_with_arg() {
    let src = "
let _result = [1, 2, 3].reduce(0) (acc, x): acc + x
";
    if let Value::Int(n) = run_src(src) {
        assert_eq!(n, 6);
    } else { panic!("expected Int(6)"); }
}

#[test]
fn test_trailing_closure_chained() {
    let src = "
let _result = [1, 2, 3].filter (x): x > 1
                        .map (x): x * 10
";
    if let Value::Array(v) = run_src(src) {
        assert_eq!(v, vec![Value::Int(20), Value::Int(30)]);
    } else { panic!("expected Array"); }
}

#[test]
fn test_trailing_closure_multiline_no_chain() {
    let src = "
let _result = [1, 2, 3].map (x):
    x * 2
";
    if let Value::Array(v) = run_src(src) {
        assert_eq!(v, vec![Value::Int(2), Value::Int(4), Value::Int(6)]);
    } else { panic!("expected Array"); }
}

#[test]
fn test_trailing_closure_multiline_chain_forbidden() {
    let src = "
let _result = [1, 2, 3].map (x):
    x * 2
.filter (x): x > 2
";
    let tokens = crate::lexer::lex(src).expect("lex");
    let result = crate::parser::parse(tokens);
    assert!(result.is_err(), "multiline trailing closure chaining should be a parse error");
}

#[test]
fn test_closure_no_paren_single_param() {
    let src = "
let f = x: x * 2
let _result = f(21)
";
    if let Value::Int(n) = run_src(src) {
        assert_eq!(n, 42);
    } else { panic!("expected Int(42)"); }
}

#[test]
fn test_trailing_closure_no_paren() {
    let src = "
let _result = [1, 2, 3].map x: x * 2
";
    if let Value::Array(v) = run_src(src) {
        assert_eq!(v, vec![Value::Int(2), Value::Int(4), Value::Int(6)]);
    } else { panic!("expected Array"); }
}

#[test]
fn test_closure_no_paren_as_arg() {
    let src = "
let _result = [1, 2, 3].filter(x: x > 1)
";
    if let Value::Array(v) = run_src(src) {
        assert_eq!(v, vec![Value::Int(2), Value::Int(3)]);
    } else { panic!("expected Array"); }
}

#[test]
fn test_trailing_no_paren_multiline() {
    let src = "
let _result = [1, 2, 3].map x:
    x * 3
";
    if let Value::Array(v) = run_src(src) {
        assert_eq!(v, vec![Value::Int(3), Value::Int(6), Value::Int(9)]);
    } else { panic!("expected Array"); }
}

#[test]
fn test_conformance_as_block() {
    let src = r#"
trait Animal:
    req string speak()

struct Dog: pass

ext Dog as Animal:
    req string speak(): "woof"

let d = Dog()
let _result = d.speak()
"#;
    assert_eq!(run_src(src), Value::Str("woof".to_string()));
}

#[test]
fn test_conformance_qualified() {
    let src = r#"
trait Animal:
    req string speak()

struct Cat:
    req string Animal.speak(): "meow"

let c = Cat()
let _result = c.speak()
"#;
    assert_eq!(run_src(src), Value::Str("meow".to_string()));
}

#[test]
fn test_conformance_missing_method() {
    let src = r#"
trait Animal:
    def string speak()
    def void walk()

struct Fish: pass

ext Fish as Animal:
    def string speak(): "..."
"#;
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    let mut interp = Interpreter::new();
    let result = interp.exec_program(&program);
    assert!(result.is_err(), "missing method should be a runtime error");
    assert!(result.unwrap_err().message.contains("walk"));
}

#[test]
fn test_conformance_header_declaration() {
    // struct Dog as Animal: — declared in header, methods in body
    let src = r#"
trait Animal:
    req string speak()

struct Dog as Animal:
    req string speak(): "woof"

let d = Dog()
let _result = d.speak()
"#;
    assert_eq!(run_src(src), Value::Str("woof".to_string()));
}

#[test]
fn test_as_decl_string() {
    let src = r#"
struct Dog:
    string name
    as string: "Dog({self.name})"

let d = Dog("Rex")
let _result = d as string
"#;
    assert_eq!(run_src(src), Value::Str("Dog(Rex)".to_string()));
}

#[test]
fn test_as_decl_named_type() {
    let src = r#"
struct Celsius:
    float value
    as float: self.value

struct Fahrenheit:
    float value
    as float: self.value

let c = Celsius(100.0)
let _result = c as float
"#;
    assert_eq!(run_src(src), Value::Float(100.0));
}

#[test]
fn test_as_decl_block() {
    let src = r#"
struct Point:
    float x
    float y
    as string:
        let xs = self.x as string
        let ys = self.y as string
        "Point(" + xs + ", " + ys + ")"

let p = Point(1.0, 2.0)
let _result = p as string
"#;
    assert_eq!(run_src(src), Value::Str("Point(1, 2)".to_string()));
}

#[test]
fn test_if_let_some() {
    let src = r#"
def string greet(string? name):
    var result = "nobody"
    if let n = name:
        result = "hello " + n
    return result

let _result = greet("Alice")
"#;
    assert_eq!(run_src(src), Value::Str("hello Alice".to_string()));
}

#[test]
fn test_if_let_nil() {
    let src = r#"
def string greet(string? name):
    if let n = name:
        return "hello " + n
    else:
        return "nobody"
    return "unreachable"

let _a = greet("Bob")
let _result = greet(nil)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_a"), Value::Str("hello Bob".to_string()));
    assert_eq!(get_var(&interp, "_result"), Value::Str("nobody".to_string()));
}

#[test]
fn test_if_let_inline() {
    let _src = r#"
let string? x = "hi"
let _result = if let s = x: s else "bye"
"#;
    // if let as inline statement (not expression, stored via assignment)
    // actually test the inline stmt form
    let src = r#"
let string? x = "hi"
var _result = "none"
if let s = x: _result = s else: _result = "bye"
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Str("hi".to_string()));
}

#[test]
fn test_optional_chaining_field() {
    let src = r#"
struct User:
    string name

let User'? u = User("Alice")
let _result = u?.name
let _nil = nil?.name
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Str("Alice".to_string()));
    assert_eq!(get_var(&interp, "_nil"), Value::Nil);
}

#[test]
fn test_optional_chaining_method() {
    let src = r#"
struct Greeter:
    def string hello(): "hello!"

let Greeter'? g = Greeter()
let _result = g?.hello()
let _nil = nil?.hello()
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Str("hello!".to_string()));
    assert_eq!(get_var(&interp, "_nil"), Value::Nil);
}

#[test]
fn test_optional_chaining_with_else() {
    let src = r#"
struct User:
    string name

let User'? u = nil
let _result = u?.name else "anonymous"
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Str("anonymous".to_string()));
}

// ─── Ownership tests ─────────────────────────────────────────────────────

#[test]
fn test_owned_param_invalidates_source() {
    // After passing a value to an owned param, the source variable must be gone
    let src = r#"
struct Dog:
    string name

def string pet(Dog' d): d.name

let d = Dog("Rex")
let _r = pet(d)
let _use_after = d
"#;
    let (_, res) = run(src);
    assert!(res.is_err(), "using a moved variable should fail");
}

#[test]
fn test_double_use_same_call_errors() {
    // Same variable passed twice to owned params → error
    let src = r#"
struct Dog:
    string name

def string feed(Dog' a, Dog' b): a.name + b.name

let d = Dog("Rex")
let _r = feed(d, d)
"#;
    let (_, res) = run(src);
    assert!(res.is_err(), "double-move in same call should fail");
}

#[test]
fn test_owned_collection_init_invalidates_sources() {
    // Variables used to initialise an [T'] collection should be invalidated
    let src = r#"
struct Dog:
    string name

let d1 = Dog("Ace")
let d2 = Dog("Bolt")
let [Dog'] pack = [d1, d2]
let _stale = d1
"#;
    let (_, res) = run(src);
    assert!(res.is_err(), "source of owned collection init should be invalidated");
}

#[test]
fn test_owned_collection_push_invalidates_source() {
    // push() to an [T'] collection should invalidate the pushed variable
    let src = r#"
struct Dog:
    string name

var [Dog'] pack = []
let d = Dog("Chase")
pack.push(d)
let _stale = d
"#;
    let (_, res) = run(src);
    assert!(res.is_err(), "source pushed into owned collection should be invalidated");
}

#[test]
fn test_task_cannot_capture_unqualified_collection() {
    // An unqualified array (no 'rc/'static/'copy/') cannot be captured by a task
    let src = r#"
struct Dog:
    string name

let [Dog'] pack = []
let f = task: pack
"#;
    let (_, res) = run(src);
    assert!(res.is_err(), "task must not capture unqualified collections");
}

#[test]
fn test_task_can_capture_owned_value() {
    // A value declared Dog' can be moved into a task (exclusive ownership)
    let src = r#"
struct Dog:
    string name

let Dog' d = Dog("Rex")
let f = task: d.name
let _result = f.value
"#;
    let (interp, res) = run(src);
    res.expect("owned value should be capturable by task");
    assert_eq!(get_var(&interp, "_result"), Value::Str("Rex".into()));
}

#[test]
fn test_task_owned_capture_invalidates_source() {
    // After a task captures a Dog', the source variable must be gone
    let src = r#"
struct Dog:
    string name

let Dog' d = Dog("Rex")
let f = task: d.name
let _stale = d
"#;
    let (_, res) = run(src);
    assert!(res.is_err(), "source of owned task capture should be invalidated");
}

// ─── Qualifier + alias tests ──────────────────────────────────────────────

#[test]
fn test_alias_declaration() {
    // User-defined alias: uppercase so it's recognised as a type name
    let src = r#"
use Kg as Float'copy
let Kg weight = 72.5
let _result = weight + 1.0
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Float(73.5));
}

#[test]
fn test_builtin_alias_int() {
    // `int` is a built-in alias for Int'copy
    let src = r#"
let int x = 42
let _result = x + 1
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(43));
}

#[test]
fn test_builtin_alias_string() {
    // `string` is a built-in alias for String'rc
    let src = r#"
let string name = "Alice"
let _result = "Hello, " + name
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Str("Hello, Alice".into()));
}

#[test]
fn test_sync_qualifier_allows_task_capture() {
    // A collection qualified 'task can be captured by a task
    let src = r#"
let [int]'task items = [1, 2, 3]
let f = task: items.len()
let _result = f.value
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(3));
}

#[test]
fn test_const_qualifier_allows_task_capture() {
    // A value qualified 'const can be captured by a task
    let src = r#"
let String'const greeting = "hello"
let f = task: greeting
let _result = f.value
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Str("hello".into()));
}

#[test]
fn test_mut_qualifier_blocks_task_capture() {
    // 'auto is not task-safe
    let src = r#"
let [int]'auto items = [1, 2, 3]
let f = task: items
"#;
    let (_, res) = run(src);
    assert!(res.is_err(), "'auto collections must not be captured by tasks");
}

#[test]
fn test_generic_future_value_in_task() {
    let src = r#"
task def int compute(): 42
let f = task compute()
let _result = f.value
"#;
    // top-level is task context, so this should work
    let (interp, res) = run(src);
    res.expect("top-level is task context");
    assert_eq!(get_var(&interp, "_result"), Value::Int(42));
}

#[test]
fn test_future_value_inside_task_fn() {
    let src = r#"
task def int compute(): 42

task def int get_result():
    let f = task compute()
    f.value

let f2 = task get_result()
let _result = f2.value
"#;
    let (interp, res) = run(src);
    res.expect("task fn can access .value");
    assert_eq!(get_var(&interp, "_result"), Value::Int(42));
}

#[test]
fn test_defer_after_normal_return() {
    let src = r#"
var log = ""

def string greet(string name):
    defer: log = log + "defer"
    log = log + "body "
    "hi " + name

let r = greet("Alice")
"#;
    let (interp, res) = run(src);
    res.expect("defer after normal return");
    assert_eq!(get_var(&interp, "r"), Value::Str("hi Alice".into()));
    assert_eq!(get_var(&interp, "log"), Value::Str("body defer".into()));
}

#[test]
fn test_defer_after_early_return() {
    let src = r#"
var log = ""

def int safe_div(int a, int b):
    defer: log = log + "defer"
    guard b != 0 else:
        log = log + "guard "
        return -1
    a / b

let r = safe_div(10, 0)
"#;
    let (interp, res) = run(src);
    res.expect("defer after early return");
    assert_eq!(get_var(&interp, "r"), Value::Int(-1));
    assert_eq!(get_var(&interp, "log"), Value::Str("guard defer".into()));
}

#[test]
fn test_defer_after_exception() {
    let src = r#"
var log = ""

def int fail() throws:
    defer: log = log + "defer"
    log = log + "body "
    throw "oops"
    0

let r = try fail() else -1
"#;
    let (interp, res) = run(src);
    res.expect("defer after exception");
    assert_eq!(get_var(&interp, "r"), Value::Int(-1));
    assert_eq!(get_var(&interp, "log"), Value::Str("body defer".into()));
}

#[test]
fn test_defer_lifo_order() {
    let src = r#"
var log = ""

def void multi():
    defer: log = log + "1"
    defer: log = log + "2"
    defer: log = log + "3"

multi()
"#;
    let (interp, res) = run(src);
    res.expect("defer LIFO");
    assert_eq!(get_var(&interp, "log"), Value::Str("321".into()));
}

#[test]
fn test_void_function() {
    let src = r#"
def void greet(string name):
    print "hi " + name

let r = greet("Alice")
"#;
    let (interp, res) = run(src);
    res.expect("void function returns Void");
    assert_eq!(get_var(&interp, "r"), Value::Void);
}

#[test]
fn test_void_literal() {
    let src = "let v = void";
    let (interp, res) = run(src);
    res.expect("void literal");
    assert_eq!(get_var(&interp, "v"), Value::Void);
}

#[test]
fn test_defer_no_colon() {
    let src = r#"
var log = ""

def void work():
    defer log = log + "defer"
    log = log + "body "

work()
"#;
    let (interp, res) = run(src);
    res.expect("defer without colon");
    assert_eq!(get_var(&interp, "log"), Value::Str("body defer".into()));
}

#[test]
fn test_defer_multiline() {
    let src = r#"
var log = ""

def void work():
    defer:
        log = log + "a"
        log = log + "b"
    log = log + "body "

work()
"#;
    let (interp, res) = run(src);
    res.expect("defer multiline");
    assert_eq!(get_var(&interp, "log"), Value::Str("body ab".into()));
}

#[test]
fn test_cast_nil_on_failure() {
    let src = r#"
let a = "42" as int
let b = "oops" as int
let c = "3.14" as float
let d = "bad" as float
let e = true as int
let f = 42 as bool
"#;
    let (interp, res) = run(src);
    res.expect("cast returns nil on failure");
    assert_eq!(get_var(&interp, "a"), Value::Int(42));
    assert_eq!(get_var(&interp, "b"), Value::Nil);
    assert_eq!(get_var(&interp, "c"), Value::Float(3.14));
    assert_eq!(get_var(&interp, "d"), Value::Nil);
    assert_eq!(get_var(&interp, "e"), Value::Int(1));
    assert_eq!(get_var(&interp, "f"), Value::Nil);
}

#[test]
fn test_tuple_destructure_no_parens() {
    let src = r#"
let t = (10, 20, 30)
let x, y, z = t
"#;
    let (interp, res) = run(src);
    res.expect("bare destructure without parens");
    assert_eq!(get_var(&interp, "x"), Value::Int(10));
    assert_eq!(get_var(&interp, "y"), Value::Int(20));
    assert_eq!(get_var(&interp, "z"), Value::Int(30));
}

#[test]
fn test_tuple_pattern_match_no_parens() {
    let src = r#"
let t = (1, 2)
var r = "none"
match t:
    1, 2: r = "one-two"
    1, _: r = "one-any"
    _:    r = "other"
"#;
    let (interp, res) = run(src);
    res.expect("tuple match without parens");
    assert_eq!(get_var(&interp, "r"), Value::Str("one-two".into()));
}

#[test]
fn test_tuple_destructure_basic() {
    let src = r#"
let t = (1, "hello", true)
let (a, b, c) = t
"#;
    let (interp, res) = run(src);
    res.expect("basic tuple destructure");
    assert_eq!(get_var(&interp, "a"), Value::Int(1));
    assert_eq!(get_var(&interp, "b"), Value::Str("hello".into()));
    assert_eq!(get_var(&interp, "c"), Value::Bool(true));
}

#[test]
fn test_tuple_destructure_wildcard() {
    let src = r#"
let t = (10, 20, 30)
let (x, _, z) = t
"#;
    let (interp, res) = run(src);
    res.expect("wildcard destructure");
    assert_eq!(get_var(&interp, "x"), Value::Int(10));
    assert_eq!(get_var(&interp, "z"), Value::Int(30));
}

#[test]
fn test_tuple_destructure_inline() {
    let src = r#"
let (p, q) = (42, "world")
"#;
    let (interp, res) = run(src);
    res.expect("inline tuple destructure");
    assert_eq!(get_var(&interp, "p"), Value::Int(42));
    assert_eq!(get_var(&interp, "q"), Value::Str("world".into()));
}

#[test]
fn test_tuple_pattern_match() {
    let src = r#"
let t = (1, 2)
var r = "none"
match t:
    (1, 2): r = "one-two"
    (1, _): r = "one-any"
    _:      r = "other"
"#;
    let (interp, res) = run(src);
    res.expect("tuple match");
    assert_eq!(get_var(&interp, "r"), Value::Str("one-two".into()));
}

#[test]
fn test_tuple_pattern_match_bind() {
    let src = r#"
let point = (3, 4)
var dist = 0
match point:
    (x, y): dist = x * x + y * y
"#;
    let (interp, res) = run(src);
    res.expect("tuple match with bind");
    assert_eq!(get_var(&interp, "dist"), Value::Int(25));
}

#[test]
fn test_default_param() {
    let src = r#"
def string greet(string name = "world"):
    "Hello, {name}!"

let a = greet("Alice")
let b = greet()
"#;
    let (interp, res) = run(src);
    res.expect("default param works");
    assert_eq!(get_var(&interp, "a"), Value::Str("Hello, Alice!".into()));
    assert_eq!(get_var(&interp, "b"), Value::Str("Hello, world!".into()));
}

#[test]
fn test_variadic_param() {
    let src = r#"
def int sum(int... args):
    var total = 0
    for n in args:
        total = total + n
    total

let s1 = sum(1, 2, 3)
let s2 = sum()
"#;
    let (interp, res) = run(src);
    res.expect("variadic param works");
    assert_eq!(get_var(&interp, "s1"), Value::Int(6));
    assert_eq!(get_var(&interp, "s2"), Value::Int(0));
}

#[test]
fn test_labeled_arg() {
    let src = r#"
def string describe(string first, string last):
    "{first} {last}"

let r1 = describe(last= "Doe", first= "John")
let r2 = describe("Jane", last= "Smith")
"#;
    let (interp, res) = run(src);
    res.expect("labeled args work");
    assert_eq!(get_var(&interp, "r1"), Value::Str("John Doe".into()));
    assert_eq!(get_var(&interp, "r2"), Value::Str("Jane Smith".into()));
}

#[test]
fn test_generic_identity_fn() {
    let src = r#"
def T identity(T x):
    x

let _a = identity(42)
let _b = identity("hello")
let _c = identity(true)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_a"), Value::Int(42));
    assert_eq!(get_var(&interp, "_b"), Value::Str("hello".into()));
    assert_eq!(get_var(&interp, "_c"), Value::Bool(true));
}

#[test]
fn test_generic_swap_fn() {
    let src = r#"
def (V, T) swap(T a, V b):
    (b, a)

let t = swap(1, "x")
let _a = t.0
let _b = t.1
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_a"), Value::Str("x".into()));
    assert_eq!(get_var(&interp, "_b"), Value::Int(1));
}

#[test]
fn test_generic_struct_fields() {
    let src = r#"
struct Pair<T, V>:
    T left
    V right

let p = Pair(10, "hello")
let _l = p.left
let _r = p.right
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_l"), Value::Int(10));
    assert_eq!(get_var(&interp, "_r"), Value::Str("hello".into()));
}

#[test]
fn test_generic_struct_method() {
    let src = r#"
struct Box<T>:
    var T value

    def T get():
        self.value

    def void set(T v):
        self.value = v

var b = Box(42)
let _before = b.get()
b.set(99)
let _after = b.get()
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_before"), Value::Int(42));
    assert_eq!(get_var(&interp, "_after"), Value::Int(99));
}

#[test]
fn test_generic_where_clause_pass() {
    // Animal implements Printable, so this should work
    let src = r#"
trait Printable:
    req string describe()

struct Animal:
    string name

    req string describe():
        self.name

def string show<T as Printable>(T item):
    item.describe()

let a = Animal("cat")
let _result = show(a)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Str("cat".into()));
}

#[test]
fn test_generic_where_clause_fail() {
    // Point does NOT implement Printable — should runtime-error
    let src = r#"
trait Printable:
    def string describe()

struct Point:
    float x
    float y

def string show<T as Printable>(T item):
    item.describe()

let p = Point(1.0, 2.0)
let _result = show(p)
"#;
    let (_interp, res) = run(src);
    assert!(res.is_err(), "expected runtime error for missing trait conformance");
}

/// `<T as (Trait1, Trait2)>` — multi-trait constraint on a single type param.
#[test]
fn test_generic_multi_trait_constraint_pass() {
    let src = r#"
trait Printable:
    req string describe()

trait Greetable:
    req string greet()

struct Person:
    string name

    req string describe():
        self.name

    req string greet():
        "Hello from {self.name}!"

def string show_and_greet<T as Printable + Greetable>(T item):
    item.describe()

let p = Person("Alice")
let _r = show_and_greet(p)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_r"), Value::Str("Alice".into()));
}

/// Multi-trait constraint fails when one trait is missing.
#[test]
fn test_generic_multi_trait_constraint_fail() {
    let src = r#"
trait Printable:
    def string describe()

trait Greetable:
    def string greet()

struct Point:
    float x
    float y
    def string describe(): "point"

def string show_and_greet<T as Printable + Greetable>(T item):
    item.describe()

let p = Point(1.0, 2.0)
let _r = show_and_greet(p)
"#;
    let (_, res) = run(src);
    assert!(res.is_err(), "expected error: Point does not implement Greetable");
}

#[test]
fn test_generic_enum_variant() {
    let src = r#"
enum Result<T, E>:
    Ok(T)
    Err(E)

let r1 = Result.Ok(42)
let r2 = Result.Err("oops")
var _ok_val = 0
var _err_val = ""
match r1:
    Ok(v): _ok_val = v
    _: _ok_val = -1
match r2:
    Err(e): _err_val = e
    _: _err_val = "?"
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_ok_val"), Value::Int(42));
    assert_eq!(get_var(&interp, "_err_val"), Value::Str("oops".into()));
}

#[test]
fn test_generic_optional_param() {
    let src = r#"
def T unwrap_or(T? opt, T default):
    opt else default

let _a = unwrap_or(nil, 42)
let _b = unwrap_or(7, 42)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_a"), Value::Int(42));
    assert_eq!(get_var(&interp, "_b"), Value::Int(7));
}

// ── Implicit type param inference (no <> required) ──────────────────────

/// Single type param inferred from parameter type — no `<T>` needed.
#[test]
fn test_implicit_type_param_identity() {
    let src = r#"
def T id(T x):
    x

let _a = id(99)
let _b = id("hi")
let _c = id(false)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_a"), Value::Int(99));
    assert_eq!(get_var(&interp, "_b"), Value::Str("hi".into()));
    assert_eq!(get_var(&interp, "_c"), Value::Bool(false));
}

/// Two distinct type params inferred from two parameters.
#[test]
fn test_implicit_type_param_two_params() {
    let src = r#"
def (T, V) pair(T a, V b):
    (a, b)

let t = pair(1, "x")
let _fst = t.0
let _snd = t.1
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_fst"), Value::Int(1));
    assert_eq!(get_var(&interp, "_snd"), Value::Str("x".into()));
}

/// Type param inferred from return type annotation only.
#[test]
fn test_implicit_type_param_from_return() {
    let src = r#"
def T? wrap(T x):
    x

let _a = wrap(7)
let _b = wrap("z")
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_a"), Value::Int(7));
    assert_eq!(get_var(&interp, "_b"), Value::Str("z".into()));
}

// ── Type-annotation enforcement ──────────────────────────────────────────

/// Declared return type is checked at runtime: wrong type raises an error.
#[test]
fn test_return_type_mismatch() {
    let src = r#"
def int bad() throws:
    "oops"

let _ = bad()
"#;
    let (_, res) = run(src);
    let err = res.expect_err("should fail with type mismatch");
    assert!(
        err.message.contains("declared to return") || err.message.contains("bad"),
        "unexpected error: {}", err.message
    );
}

/// Declared return type is satisfied: no error.
#[test]
fn test_return_type_ok() {
    let src = r#"
def int good():
    42

let x = good()
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "x"), Value::Int(42));
}

/// Optional return type: nil is valid.
#[test]
fn test_return_type_optional_nil_ok() {
    let src = r#"
def int? maybe(bool flag):
    if flag: 1 else nil

let a = maybe(true)
let b = maybe(false)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "a"), Value::Int(1));
    assert_eq!(get_var(&interp, "b"), Value::Nil);
}

/// Argument type mismatch is caught at call time.
#[test]
fn test_param_type_mismatch() {
    let src = r#"
def int double(int n) throws:
    n * 2

let _ = double("hello")
"#;
    let (_, res) = run(src);
    let err = res.expect_err("should fail with param type mismatch");
    assert!(
        err.message.contains("argument 'n'") || err.message.contains("expected int"),
        "unexpected error: {}", err.message
    );
}

/// Correct argument type: no error.
#[test]
fn test_param_type_ok() {
    let src = r#"
def int double(int n):
    n * 2

let x = double(21)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "x"), Value::Int(42));
}

/// uint coercion is applied to both param and return value.
#[test]
fn test_uint_coercion_param_and_return() {
    let src = r#"
def uint pass_uint(uint n):
    n

let x = pass_uint(7)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "x"), Value::Uint(7));
}

#[test]
fn test_getter_basic() {
    let src = r#"
struct Dog:
    string _name
    req string name(): self._name

let d = Dog("Rex")
let _n = d.name
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_n"), Value::Str("Rex".into()));
}

#[test]
fn test_setter_basic() {
    let src = r#"
struct Dog:
    var string _name
    req string name(): self._name
    set name(string val): self._name = val

var d = Dog("Rex")
d.name = "Buddy"
let _n = d.name
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_n"), Value::Str("Buddy".into()));
}

#[test]
fn test_getter_computed() {
    let src = r#"
struct Circle:
    float radius
    req float area(): 3.14159 * self.radius * self.radius

let c = Circle(2.0)
let _a = c.area
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    // area = 3.14159 * 2.0 * 2.0 = 12.56636
    if let Value::Float(f) = get_var(&interp, "_a") {
        assert!((f - 12.56636).abs() < 0.001, "expected ~12.566, got {}", f);
    } else {
        panic!("expected Float, got {:?}", get_var(&interp, "_a"));
    }
}

#[test]
fn test_conformance_multi_header() {
    // struct Dog as Animal, Printable: — multiple protocols in header
    let src = r#"
trait Animal:
    req string speak()

trait Printable:
    req string print()

struct Dog as Animal, Printable:
    string name
    req string speak(): "woof"
    req string print(): "Dog(" + self.name + ")"

let d = Dog("Rex")
let _result = d.speak() + "|" + d.print()
"#;
    assert_eq!(run_src(src), Value::Str("woof|Dog(Rex)".to_string()));
}

#[test]
fn test_struct_composition_fields() {
    // Dog composes Animal, exposes its fields via composition
    let src = r#"
struct Animal:
    init(pub string name)

struct Dog:
    init(pub Animal base, pub string breed)
    req string describe(): self.base.name + " (" + self.breed + ")"

let d = Dog(base = Animal(name = "Rex"), breed = "Labrador")
let _result = d.describe()
"#;
    assert_eq!(run_src(src), Value::Str("Rex (Labrador)".to_string()));
}

#[test]
fn test_struct_as_conversion() {
    // `as Animal:` defines an explicit conversion from Dog to Animal
    let src = r#"
struct Animal:
    init(pub string name)

struct Dog:
    init(pub Animal base, pub string breed)
    as Animal:
        self.base

let d = Dog(base = Animal(name = "Rex"), breed = "Labrador")
let a = d as Animal
let _result = a.name
"#;
    assert_eq!(run_src(src), Value::Str("Rex".to_string()));
}

#[test]
fn test_struct_as_conversion_in_fn() {
    // `as Animal:` conversion used implicitly at a call site
    let src = r#"
struct Animal:
    init(pub string name)

struct Dog:
    init(pub Animal base, pub string breed)
    as Animal:
        self.base

def string greet(Animal a): "Hello " + a.name

let d = Dog(base = Animal(name = "Rex"), breed = "Labrador")
let _result = greet(d as Animal)
"#;
    assert_eq!(run_src(src), Value::Str("Hello Rex".to_string()));
}

#[test]
fn test_struct_as_implicit_coercion_fn_call() {
    // Passing Dog where Animal is expected — implicit `as Animal:` conversion at call site
    let src = r#"
struct Animal:
    init(pub string name)

struct Dog:
    init(pub Animal base, pub string breed)
    as Animal:
        self.base

def string greet(Animal a): "Hello " + a.name

let d = Dog(base = Animal(name = "Rex"), breed = "Labrador")
let _result = greet(d)
"#;
    assert_eq!(run_src(src), Value::Str("Hello Rex".to_string()));
}

#[test]
fn test_struct_as_implicit_coercion_let() {
    // `let Animal a = dog` triggers implicit `as Animal:` conversion
    let src = r#"
struct Animal:
    init(pub string name)

struct Dog:
    init(pub Animal base, pub string breed)
    as Animal:
        self.base

let d = Dog(base = Animal(name = "Buddy"), breed = "Husky")
let Animal a = d
let _result = a.name
"#;
    assert_eq!(run_src(src), Value::Str("Buddy".to_string()));
}

#[test]
fn test_struct_composition_with_trait() {
    // Dog composes Animal, also conforms to Printable
    let src = r#"
struct Animal:
    init(pub string name)

trait Printable:
    req string print()

struct Dog as Printable:
    init(pub Animal base)
    req string print(): "Dog:" + self.base.name

let d = Dog(base = Animal(name = "Rex"))
let _result = d.print()
"#;
    assert_eq!(run_src(src), Value::Str("Dog:Rex".to_string()));
}

// ─── Specialized generics ────────────────────────────────────────────────────

#[test]
fn test_collection_type_annotation_native() {
    // [int], {string = int}, {int} are the canonical annotation forms
    let src = r#"
def int sum([int] nums):
    var s = 0
    for n in nums:
        s = s + n
    return s

let _result = sum([1, 2, 3])
"#;
    assert_eq!(run_src(src), Value::Int(6));
}

#[test]
fn test_generic_future_inner_type() {
    // Future<int> — inner value must match
    let src = r#"
task def int get():
    99

let f = task get()
let _result = f.value
"#;
    assert_eq!(run_src(src), Value::Int(99));
}

#[test]
fn test_generic_struct_field_check() {
    // User-defined generic struct: Stack<T>
    let src = r#"
struct Stack<T>:
    T item

let s = Stack(42)
let _result = s.item
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_generic_struct_specialized_param() {
    // Passing a Stack<int> where Stack<int>'task is expected
    let src = r#"
struct Stack<T>:
    T item
    req T get(): self.item

def int unwrap(Stack<int>'task s): s.get()

let s = Stack(21)
let _result = unwrap(s)
"#;
    assert_eq!(run_src(src), Value::Int(21));
}

// ─── ext tests ───────────────────────────────────────────────────────────────

#[test]
fn test_ext_adds_method() {
    let src = r#"
struct Dog:
    string name

ext Dog:
    req string greet(): "Woof, I am {self.name}!"

let d = Dog(name = "Rex")
let _result = d.greet()
"#;
    assert_eq!(run_src(src), Value::Str("Woof, I am Rex!".into()));
}

#[test]
fn test_ext_overrides_method() {
    let src = r#"
struct Dog:
    req string speak(): "woof"

ext Dog:
    req string speak(): "WOOF"

let d = Dog()
let _result = d.speak()
"#;
    assert_eq!(run_src(src), Value::Str("WOOF".into()));
}

#[test]
fn test_ext_as_conformance() {
    let src = r#"
trait Printable:
    req string describe()

struct Dog:
    string name

ext Dog as Printable:
    req string describe(): "Dog({self.name})"

def string print_it(Printable'task p): p.describe()

let d = Dog(name = "Fido")
let _result = print_it(d)
"#;
    assert_eq!(run_src(src), Value::Str("Dog(Fido)".into()));
}

#[test]
fn test_ext_multiple_traits() {
    let src = r#"
trait Named:
    req string name()

trait Aged:
    req int age()

struct Cat:
    string cat_name
    int cat_age

ext Cat as Named, Aged:
    req string name(): self.cat_name
    req int age(): self.cat_age

def string describe(Named'task n, Aged'task a): "{n.name()} is {a.age()}"

let c = Cat(cat_name = "Mimi", cat_age = 3)
let _result = describe(c, c)
"#;
    assert_eq!(run_src(src), Value::Str("Mimi is 3".into()));
}

#[test]
fn test_ext_missing_trait_method_errors() {
    let src = r#"
trait Printable:
    def string describe()

struct Dog:
    string name

ext Dog as Printable:
    def string bark(): "woof"
"#;
    let (_interp, result) = run(src);
    assert!(result.is_err());
}


#[test]
fn test_ext_as_conversion() {
    let src = r#"
struct Celsius:
    float degrees

ext Celsius:
    as float:
        return self.degrees
    as string:
        return "{self.degrees}°C"

let c = Celsius(degrees= 100.0)
let f = c as float
let s = c as string
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "f"), Value::Float(100.0));
    assert_eq!(get_var(&interp, "s"), Value::Str("100°C".into()));
}

#[test]
fn test_ext_getter_setter() {
    let src = r#"
struct Counter:
    var int _value

ext Counter:
    req int value(): self._value
    set value(int v):
        self._value = v

var c = Counter(_value= 0)
c.value = 42
let _result = c.value
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_field_mutability() {
    // var field: assignable
    let src_ok = r#"
struct Point:
    var int x
    var int y

var p = Point(x = 1, y = 2)
p.x = 10
let _result = p.x
"#;
    assert_eq!(run_src(src_ok), Value::Int(10));

    // explicit `let` field: not assignable, private
    let src_let = r#"
struct Pair:
    let int a

var p = Pair(a= 1)
p.a = 99
"#;
    let (_i, res) = run(src_let);
    assert!(res.is_err(), "assigning to explicit-let field should error");

    // implicit field (no keyword): pub let — not assignable
    let src_implicit = r#"
struct Pair:
    int a

var p = Pair(a= 1)
p.a = 99
"#;
    let (_i, res) = run(src_implicit);
    assert!(res.is_err(), "assigning to implicit pub-let field should error");
}

#[test]
fn test_ext_pub() {
    let src = r#"
struct Point:
    float x
    float y

ext Point:
    pub def float magnitude():
        (self.x * self.x + self.y * self.y) as float
    pub req string label(): "({self.x}, {self.y})"
    pub as string: "Point({self.x}, {self.y})"

let p = Point(x = 3.0, y = 4.0)
let _label = p.label
let _str = p as string
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_label"), Value::Str("(3, 4)".into()));
    assert_eq!(get_var(&interp, "_str"), Value::Str("Point(3, 4)".into()));
}

#[test]
fn test_enum_method() {
    let src = r#"
enum Color:
    Red
    Green
    Blue

    def string name():
        match self:
            Red: "red"
            Green: "green"
            Blue: "blue"

let c = Color.Red
let _result = c.name()
"#;
    assert_eq!(run_src(src), Value::Str("red".into()));
}

#[test]
fn test_enum_getter() {
    let src = r#"
enum Direction:
    North
    South
    East
    West

    req string label():
        match self:
            North: "nord"
            South: "sud"
            East: "est"
            _: "ouest"

let d = Direction.South
let _result = d.label
"#;
    assert_eq!(run_src(src), Value::Str("sud".into()));
}


mod tests_part2;
mod tests_part3;
