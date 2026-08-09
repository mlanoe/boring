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
fn test_stack_qualifier_return() {
    // 'stack as return type qualifier
    let src = r#"
struct Pair:
    int a
    int b

def Pair'stack make(int a, int b):
    return Pair(a, b)

let p = make(20, 22)
let _result = p.a + p.b
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

// ─── Point 4: Numeric type aliases ──────────────────────────────────────────

#[test]
fn test_numeric_i32_alias() {
    // i32 is a distinct real 32-bit type at runtime (Value::Int32), not an int alias.
    let src = r#"
def i32 add(i32 x, i32 y):
    return x + y

let _result = add(10, 32)
"#;
    assert_eq!(run_src(src), Value::Int32(42));
}

#[test]
fn test_numeric_u64_alias() {
    // u64 is a distinct real 64-bit type at runtime (Value::Uint64), not a uint alias.
    // Mixing with a bare int literal (`2`) is allowed — literals are the flexible Int kind.
    let src = r#"
def u64 double(u64 x):
    return x * 2

let _result = double(21)
"#;
    assert_eq!(run_src(src), Value::Uint64(42));
}

#[test]
fn test_numeric_f32_alias() {
    // f32 is a real, distinct 32-bit float type at runtime — Value::Float32,
    // not a label collapsing into Value::Float64 (docs/float-width-types.md).
    let src = r#"
def f32 half(f32 x):
    return x / 2.0

let _result = half(3.14)
"#;
    let Value::Float32(v) = run_src(src) else { panic!("expected float32") };
    assert!((v - 1.57).abs() < 1e-5);
}

#[test]
fn test_numeric_usize_alias() {
    // usize is an alias for uint'copy at runtime
    let src = r#"
let usize n = 10
let usize _result = n + 5
"#;
    assert_eq!(run_src(src), Value::Uint(15));
}

#[test]
fn test_numeric_isize_alias() {
    // isize is an alias for int'copy at runtime
    let src = r#"
let isize n = -10
let isize _result = n + 5
"#;
    assert_eq!(run_src(src), Value::Int(-5));
}

#[test]
fn test_numeric_all_int_aliases() {
    // i8, i16, i32, i64 are distinct real fixed-width types at runtime; isize is the
    // bare `int` alias (unlike the old collapsing behavior where all of these acted as int).
    let src = r#"
let i8 a = 1
let i16 b = 2
let i32 c = 3
let i64 d = 4
let isize e = 5
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "a"), Value::Int8(1));
    assert_eq!(get_var(&interp, "b"), Value::Int16(2));
    assert_eq!(get_var(&interp, "c"), Value::Int32(3));
    assert_eq!(get_var(&interp, "d"), Value::Int64(4));
    assert_eq!(get_var(&interp, "e"), Value::Int(5));
}

#[test]
fn test_numeric_all_uint_aliases() {
    // u8, u16, u32, u64 are distinct real fixed-width types at runtime; usize is the
    // bare `uint` alias (unlike the old collapsing behavior where all of these acted as uint).
    let src = r#"
let u8 a = 1
let u16 b = 2
let u32 c = 3
let u64 d = 4
let usize e = 5
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "a"), Value::Uint8(1));
    assert_eq!(get_var(&interp, "b"), Value::Uint16(2));
    assert_eq!(get_var(&interp, "c"), Value::Uint32(3));
    assert_eq!(get_var(&interp, "d"), Value::Uint64(4));
    assert_eq!(get_var(&interp, "e"), Value::Uint(5));
}

#[test]
fn test_numeric_fixed_width_same_kind_arithmetic() {
    // Same-kind arithmetic works natively at each width, including the 128-bit kinds.
    let src = r#"
let i32 a = 1000000000
let i32 b = 500000000
let sum32 = a + b
let u128 c = 1
let u128 d = 2
let sum128 = c + d
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "sum32"), Value::Int32(1500000000));
    assert_eq!(get_var(&interp, "sum128"), Value::Uint128(3));
}

#[test]
fn test_numeric_fixed_width_mixes_with_bare_int_literal() {
    // A fixed-width value mixes freely with the flexible bare `Int`/`Uint` literal kind
    // (the common case: `counter + 1`), widening/narrowing per the wider-kind-wins rule.
    let src = r#"
let uint32 count = 10
let bigger = count + 1
let int8 small = 5
let smaller = small - 2
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    // `count`(32-bit) is wider than the bare literal(would-be 64-bit only if bare were
    // narrower) — here bare `Int` (64-bit) is wider than Uint32, so the bare kind wins.
    assert_eq!(get_var(&interp, "bigger"), Value::Int(11));
    assert_eq!(get_var(&interp, "smaller"), Value::Int(3));
}

#[test]
fn test_numeric_distinct_fixed_width_mix_is_error() {
    // Mixing two *different* explicit fixed-width kinds directly is a type error —
    // mirrors Rust's own refusal to implicitly coerce between distinct integer types.
    let src = r#"
let uint16 a = 1
let int32 b = 2
let _result = a + b
"#;
    let (_interp, res) = run(src);
    let err = res.expect_err("expected a runtime error mixing Uint16 and Int32 directly");
    assert!(err.message.contains("cannot add"), "unexpected error: {}", err.message);
}

#[test]
fn test_numeric_fixed_width_cast_range_check() {
    // Casting to a narrower fixed-width type checks the range, same as the existing
    // `uint8` behavior — an out-of-range value casts to nil rather than wrapping.
    let src = r#"
let int8 in_range = 100 as int8
let out_of_range = 300 as int8
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "in_range"), Value::Int8(100));
    assert_eq!(get_var(&interp, "out_of_range"), Value::Nil);
}

#[test]
fn test_numeric_fixed_width_cross_cast() {
    // `as` is the escape hatch for mixing two distinct fixed-width kinds: cast one
    // explicitly to the other's kind (or to the flexible bare kind) first.
    let src = r#"
let uint16 a = 100
let int32 b = 5
let _result = (a as int32) + b
"#;
    assert_eq!(run_src(src), Value::Int32(105));
}

// ─── Point 5: Module declarations ───────────────────────────────────────────

#[test]
fn test_mod_defines_items_in_scope() {
    // Items inside mod: are accessible in the enclosing scope (flat scoping)
    let src = r#"
mod utils:
    def int double(int x):
        return x * 2

let _result = double(21)
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_mod_struct_accessible() {
    let src = r#"
mod geometry:
    struct Point:
        int x
        int y

let p = Point(3, 4)
let _result = p.x + p.y
"#;
    assert_eq!(run_src(src), Value::Int(7));
}

#[test]
fn test_mod_nested() {
    let src = r#"
mod outer:
    mod inner:
        def int val():
            return 42

let _result = val()
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_mod_multiple_fns() {
    let src = r#"
mod math:
    def int add(int x, int y): return x + y
    def int mul(int x, int y): return x * y

let _result = add(6, mul(4, 9))
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

// ─── Point 6: any Trait / Trait-name types ───────────────────────────────────

#[test]
fn test_impl_trait_return_transparent() {
    // Trait name used directly as return type — no `impl` keyword needed
    let src = r#"
trait Greet:
    req string greet()

struct Hello: pass

ext Hello as Greet:
    req string greet(): "hello"

def Greet make_greeter():
    return Hello()

let g = make_greeter()
let _result = g.greet()
"#;
    assert_eq!(run_src(src), Value::Str("hello".into()));
}

#[test]
fn test_impl_trait_in_param() {
    // Trait name used directly as parameter type — no `impl` keyword needed
    let src = r#"
trait Addable:
    req int get_val()

struct Num:
    int n

ext Num as Addable:
    req int get_val(): self.n

def int extract(Addable x):
    return x.get_val()

let n = Num(42)
let _result = extract(n)
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

// ─── Point 7: Operator overloading ──────────────────────────────────────────

#[test]
fn test_operator_overload_add() {
    let src = r#"
struct Vec2:
    float x
    float y

ext Vec2:
    def Vec2' add(Vec2' rhs):
        return Vec2(self.x + rhs.x, self.y + rhs.y)

let a = Vec2(1.0, 2.0)
let b = Vec2(3.0, 4.0)
let c = a + b
let _result = c.x
"#;
    assert_eq!(run_src(src), Value::Float64(4.0));
}

#[test]
fn test_operator_overload_add_y() {
    let src = r#"
struct Vec2:
    float x
    float y

ext Vec2:
    def Vec2' add(Vec2' rhs):
        return Vec2(self.x + rhs.x, self.y + rhs.y)

let a = Vec2(1.0, 2.0)
let b = Vec2(3.0, 4.0)
let c = a + b
let _result = c.y
"#;
    assert_eq!(run_src(src), Value::Float64(6.0));
}

#[test]
fn test_operator_overload_sub() {
    let src = r#"
struct Vec2:
    float x
    float y

ext Vec2:
    def Vec2' sub(Vec2' rhs):
        return Vec2(self.x - rhs.x, self.y - rhs.y)

let a = Vec2(5.0, 8.0)
let b = Vec2(3.0, 4.0)
let c = a - b
let _result = c.x
"#;
    assert_eq!(run_src(src), Value::Float64(2.0));
}

#[test]
fn test_operator_overload_mul() {
    let src = r#"
struct Vec2:
    float x
    float y

ext Vec2:
    def Vec2' mul(float rhs):
        return Vec2(self.x * rhs, self.y * rhs)

let a = Vec2(2.0, 3.0)
let b = a * 4.0
let _result = b.x
"#;
    assert_eq!(run_src(src), Value::Float64(8.0));
}

#[test]
fn test_operator_overload_neg() {
    let src = r#"
struct Vec2:
    float x
    float y

ext Vec2:
    def Vec2' neg():
        return Vec2(-self.x, -self.y)

let a = Vec2(1.0, 2.0)
let b = -a
let _result = b.x
"#;
    assert_eq!(run_src(src), Value::Float64(-1.0));
}

#[test]
fn test_operator_overload_eq() {
    let src = r#"
struct Vec2:
    float x
    float y

ext Vec2:
    req bool eq(Vec2' rhs):
        return self.x == rhs.x and self.y == rhs.y

let a = Vec2(1.0, 2.0)
let b = Vec2(1.0, 2.0)
let c = Vec2(3.0, 4.0)
let _result = a == b
"#;
    assert_eq!(run_src(src), Value::Bool(true));
}

#[test]
fn test_operator_overload_eq_false() {
    let src = r#"
struct Vec2:
    float x
    float y

ext Vec2:
    req bool eq(Vec2' rhs):
        return self.x == rhs.x and self.y == rhs.y

let a = Vec2(1.0, 2.0)
let c = Vec2(3.0, 4.0)
let _result = a == c
"#;
    assert_eq!(run_src(src), Value::Bool(false));
}

#[test]
fn test_operator_overload_ne_via_eq() {
    let src = r#"
struct Vec2:
    float x
    float y

ext Vec2:
    req bool eq(Vec2' rhs):
        return self.x == rhs.x and self.y == rhs.y

let a = Vec2(1.0, 2.0)
let c = Vec2(3.0, 4.0)
let _result = a != c
"#;
    assert_eq!(run_src(src), Value::Bool(true));
}

#[test]
fn test_operator_overload_lt() {
    let src = r#"
struct Wrapper:
    int val

ext Wrapper:
    req bool lt(Wrapper' rhs):
        return self.val < rhs.val

let a = Wrapper(3)
let b = Wrapper(5)
let _result = a < b
"#;
    assert_eq!(run_src(src), Value::Bool(true));
}

#[test]
fn test_operator_overload_rem() {
    let src = r#"
struct Mod:
    int n

ext Mod:
    def Mod' rem(Mod' rhs):
        return Mod(self.n % rhs.n)

let a = Mod(10)
let b = Mod(3)
let c = a % b
let _result = c.n
"#;
    assert_eq!(run_src(src), Value::Int(1));
}

// ─── Point 8: @derive attributes ────────────────────────────────────────────

#[test]
fn test_derive_attr_on_struct_parsed() {
    // @derive attributes should parse without error and be accessible on the AST
    let src = r#"
@derive(Debug, Clone)
struct Point:
    int x
    int y

let p = Point(1, 2)
let _result = p.x + p.y
"#;
    assert_eq!(run_src(src), Value::Int(3));
}

#[test]
fn test_derive_attr_on_struct_multiple() {
    // Multiple stacked attributes
    let src = r#"
@derive(Debug)
@derive(Clone, PartialEq)
struct Config:
    int value

let c = Config(42)
let _result = c.value
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_derive_attr_on_fn() {
    // @inline attribute on a function
    let src = r#"
@inline
def int double(int x):
    return x * 2

let _result = double(21)
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_derive_attr_on_enum() {
    // @derive attribute on an enum
    let src = r#"
@derive(Debug, Clone)
enum Color:
    Red
    Green
    Blue

let c = Color.Red
let _result = 42
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_derive_attr_key_value_args() {
    // @serde(rename_all = "camelCase") style attribute
    let src = r#"
@serde(rename_all = "camelCase")
struct Config:
    int maxRetries

let c = Config(3)
let _result = c.maxRetries
"#;
    assert_eq!(run_src(src), Value::Int(3));
}

#[test]
fn test_attr_ast_name_preserved() {
    // Check that attr name is stored in the AST
    let src = "@derive(Debug)\nstruct Foo: pass\n";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let crate::ast::Item::Struct(decl) = &program.items[0] {
        assert_eq!(decl.attrs.len(), 1);
        assert_eq!(decl.attrs[0].name, "derive");
        assert_eq!(decl.attrs[0].args, vec!["Debug"]);
    } else {
        panic!("expected Struct item");
    }
}

#[test]
fn test_derive_attr_before_pub_struct_is_not_dropped() {
    // Regression test: `@derive(...)` on the line before `pub struct` used to
    // be silently discarded — the parser only recognized Def/Req/Struct/Enum
    // immediately after an attribute list, so a `Pub` token in between fell
    // through to a catch-all that reparsed the item with no attributes at
    // all (no error). `pub @derive(...)` (pub first) already worked; this
    // covers the previously-broken ordering.
    let src = "@derive(Debug)\npub struct Foo:\n    int x\n";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let crate::ast::Item::Struct(decl) = &program.items[0] {
        assert!(decl.is_pub, "`pub` after the attribute list should still mark the struct pub");
        assert_eq!(decl.attrs.len(), 1, "the @derive attribute must not be dropped");
        assert_eq!(decl.attrs[0].name, "derive");
        assert_eq!(decl.attrs[0].args, vec!["Debug"]);
    } else {
        panic!("expected Struct item");
    }
}

#[test]
fn test_derive_attr_after_pub_fn_is_not_dropped() {
    // Same regression, function form: `@inline` before `pub def`.
    let src = "@inline\npub def int double(int x):\n    return x * 2\n";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let crate::ast::Item::Fn(decl) = &program.items[0] {
        assert!(decl.is_pub);
        assert_eq!(decl.attrs.len(), 1);
        assert_eq!(decl.attrs[0].name, "inline");
    } else {
        panic!("expected Fn item");
    }
}

#[test]
fn test_attr_before_mod_still_parses_with_attrs_discarded() {
    // Non-regression: `@cfg(test)` before `mod` (examples/todo.br's real
    // pattern) has no AST slot to attach attributes to — must keep parsing
    // successfully with the attribute discarded, not become a hard error.
    let src = "@cfg(test)\nmod tests:\n    let x = 1\n";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_attr_ast_multiple_args() {
    // Check that multiple args are stored
    let src = "@derive(Debug, Clone, PartialEq)\nstruct Foo: pass\n";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let crate::ast::Item::Struct(decl) = &program.items[0] {
        assert_eq!(decl.attrs[0].args, vec!["Debug", "Clone", "PartialEq"]);
    } else {
        panic!("expected Struct item");
    }
}

// ─── Enum bug fixes ──────────────────────────────────────────────────────────

// Bug 1: Named field access on enum variants

#[test]
fn test_enum_variant_named_field_access() {
    let src = r#"
enum Pair:
    Both(int first, int second)

let p = Pair.Both(10, 32)
let _result = p.first + p.second
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_enum_variant_single_named_field() {
    let src = r#"
enum Wrapper:
    Value(int n)

let w = Wrapper.Value(42)
let _result = w.n
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

// Bug 2: Dot-prefix shorthand .Variant

#[test]
fn test_dot_prefix_unit_variant() {
    let src = r#"
enum Color:
    Red
    Green
    Blue

def string name(Color' c):
    match c:
        Red: "red"
        Green: "green"
        Blue: "blue"

let _result = name(.Red)
"#;
    assert_eq!(run_src(src), Value::Str("red".into()));
}

#[test]
fn test_dot_prefix_in_let() {
    let src = r#"
enum Dir:
    North
    South

let d = .North
var _result = 0
match d:
    North: _result = 1
    South: _result = 2
"#;
    assert_eq!(run_src(src), Value::Int(1));
}

// Bug 3 (or-patterns): existing feature verification

#[test]
fn test_or_pattern_enum_variants() {
    let src = r#"
enum Color:
    Red
    Green
    Blue

let c = Color.Red
let _result = match c:
    Red | Blue: "warm"
    Green: "cool"
"#;
    assert_eq!(run_src(src), Value::Str("warm".into()));
}

// Nested variant patterns

#[test]
fn test_nested_enum_pattern() {
    let src = r#"
enum Inner:
    A(int x)
    B

enum Outer:
    Wrap(Inner' v)
    Empty

let o = Outer.Wrap(Inner.A(42))
let _result = match o:
    Wrap(A(n)): n
    _: 0
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

// Alias use case: `use X as T'shared` then `def foo(X& x)` — the canonical pattern.
#[test]
fn test_postfix_borrow_via_alias() {
    let src = r#"
struct Tree:
    init(int value)

use Node as Tree'shared

def int read(Node& n):
    n.value

let Node root = Tree(42)
let _result = read(root)
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

// ─── T = stack by default, T& = borrow ───────────────────────────────────────

// Bare `T` in a param = stack-owned (no qualifier needed). Rust default.
#[test]
fn test_bare_type_param_stack_owned() {
    let src = r#"
struct Point:
    init(pub int x, pub int y)

def int sum(Point p):
    p.x + p.y

let p = Point(3, 4)
let _result = sum(p)
"#;
    assert_eq!(run_src(src), Value::Int(7));
}

// `T&` in a param = borrow (same runtime semantics as T for the interpreter,
// but the Rust transpiler will emit `&T`).
#[test]
fn test_ref_type_param_borrow() {
    let src = r#"
struct Point:
    init(pub int x, pub int y)

def int sum(Point& p):
    p.x + p.y

let p = Point(3, 4)
let _result = sum(p)
"#;
    assert_eq!(run_src(src), Value::Int(7));
}

// Bare `T` in a let annotation = stack-owned.
#[test]
fn test_bare_type_let_annotation() {
    let src = r#"
struct Counter:
    init(pub int n)

let Counter c = Counter(42)
let _result = c.n
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

// `T&` vs `T'` (borrow vs heap-owned): both accepted, different Rust output.
#[test]
fn test_borrow_vs_owned_annotations() {
    let src = r#"
struct Item:
    init(pub int v)

def int read_stack(Item i): i.v
def int read_borrow(Item& i): i.v
def int read_heap(Item' i): i.v

let Item i = Item(10)
let _result = read_stack(i) + read_borrow(i) + read_heap(i)
"#;
    assert_eq!(run_src(src), Value::Int(30));
}

// ─── Type alias as constructor ────────────────────────────────────────────────

// `use Dog2 as Dog'stack` then `Dog2("rex")` as constructor call.
#[test]
fn test_alias_constructor_stack() {
    let src = r#"
struct Dog:
    init(pub string name)

use Dog2 as Dog'stack
let d = Dog2("rex")
let _result = d.name
"#;
    assert_eq!(run_src(src), Value::Str("rex".into()));
}

// Alias for heap variant works the same way.
#[test]
fn test_alias_constructor_heap() {
    let src = r#"
struct Point:
    init(pub int x, pub int y)

use P as Point'
let p = P(3, 4)
let _result = p.x + p.y
"#;
    assert_eq!(run_src(src), Value::Int(7));
}

// Alias for 'shared variant — constructor creates the same object.
#[test]
fn test_alias_constructor_auto() {
    let src = r#"
struct Counter:
    init(pub int n)

use SharedCounter as Counter'shared
let c = SharedCounter(10)
let _result = c.n
"#;
    assert_eq!(run_src(src), Value::Int(10));
}

// Alias used with the borrow syntax: `use Node as Tree'shared; def walk(Node& n)`.
#[test]
fn test_alias_constructor_and_borrow() {
    let src = r#"
struct Tree:
    init(pub int value)

use Node as Tree'shared

def int read(Node& n):
    n.value

let n = Node(99)
let _result = read(n)
"#;
    assert_eq!(run_src(src), Value::Int(99));
}

// ── var T& — mutable borrow (&mut T) ──────────────────────────────────────────

// `var T&` as a parameter — function receives a mutable reference and modifies via it.
#[test]
fn test_borrow_mut_param() {
    let src = r#"
struct Counter:
    init(pub var int n)

def void increment(var Counter& c):
    c.n = c.n + 1

var c = Counter(0)
increment(c)
increment(c)
let _result = c.n
"#;
    assert_eq!(run_src(src), Value::Int(2));
}

// `var mut T&` in a let binding annotation — accepted without error.
// (`var T&` alone no longer implies content mutation — docs/mut-type-modifier.md
// §1/§2 — bare `mut T&` is used below since `ref` is never reassigned here.)
#[test]
fn test_borrow_mut_let_annotation() {
    let src = r#"
struct Point:
    init(pub var int x, pub var int y)

var p = Point(3, 4)
mut Point& ref = p
ref.x = 10
let _result = p.x
"#;
    assert_eq!(run_src(src), Value::Int(10));
}

// Mixing `T&` (immutable) and `var T&` (mutable) params — both accepted.
#[test]
fn test_borrow_immut_and_mut_params() {
    let src = r#"
struct Val:
    init(pub var int n)

def void copy_into(Val& src, var Val& dst):
    dst.n = src.n

var a = Val(42)
var b = Val(0)
copy_into(a, b)
let _result = b.n
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

// ── Must-use return values ──────────────────────────────────────────────────

// Void function called as a statement — OK, no value to discard.
#[test]
fn test_must_use_void_stmt_ok() {
    let src = r#"
def void say():
    print "hi"

say()
"#;
    let (_, res) = run(src);
    res.expect("void function as statement is always ok");
}

// Non-void function called as a bare statement — must-use error.
#[test]
fn test_must_use_bare_call_err() {
    let src = r#"
def int answer():
    42

answer()
"#;
    let (_, res) = run(src);
    assert!(res.is_err(), "bare non-void call should be a must-use error");
    let msg = res.unwrap_err().message;
    assert!(msg.contains("return value discarded"), "unexpected error: {msg}");
}

// `_ = f()` — explicit discard, no error.
#[test]
fn test_must_use_discard_ok() {
    let src = r#"
def int answer():
    42

_ = answer()
"#;
    let (_, res) = run(src);
    res.expect("_ = f() is an explicit discard — should succeed");
}

// `let x = f()` — bound return value, no error.
#[test]
fn test_must_use_bound_ok() {
    let src = r#"
def int double(int n):
    n * 2

let _result = double(21)
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

// Non-void method call as bare statement — must-use error.
// `req` = non-mutating read method, callable on a `let` binding.
#[test]
fn test_must_use_bare_method_call_err() {
    let src = r#"
struct Wrapper:
    init(pub int val)
    req int peek(): self.val

let w = Wrapper(7)
w.peek()
"#;
    let (_, res) = run(src);
    assert!(res.is_err(), "bare non-void method call should be a must-use error");
    let msg = res.unwrap_err().message;
    assert!(msg.contains("return value discarded"), "unexpected error: {msg}");
}

// `_ = obj.method()` — explicit discard for method call, no error.
#[test]
fn test_must_use_method_discard_ok() {
    let src = r#"
struct Wrapper:
    init(pub int val)
    req int peek(): self.val

let w = Wrapper(7)
_ = w.peek()
"#;
    let (_, res) = run(src);
    res.expect("_ = obj.method() is an explicit discard — should succeed");
}

// ── do: scoped block ────────────────────────────────────────────────────────

// Basic scoped block — local vars don't leak into outer scope.
#[test]
fn test_do_block_local_scope() {
    let src = r#"
var outer = 0
do:
    var inner = 42
    outer = inner
let _result = outer
"#;
    assert_eq!(run_src(src), Value::Int(42));
    // `inner` must not be visible outside — verify by running a separate check
    let src2 = r#"
do:
    var x = 1
let _result = 0
"#;
    let (_, res) = run(src2);
    res.expect("do: block with local var compiles and runs");
}

// `let v = do:` — block as expression; last expr is the value.
#[test]
fn test_do_block_as_expression() {
    let src = r#"
let _result = do:
    var a = 3
    var b = 4
    a + b
"#;
    assert_eq!(run_src(src), Value::Int(7));
}

// `do:` as expression with assignment on the left.
#[test]
fn test_do_block_assigned() {
    let src = r#"
var _result = do:
    10 * 10
"#;
    assert_eq!(run_src(src), Value::Int(100));
}

// `defer` inside `do:` runs when the block exits, not at end of function.
#[test]
fn test_do_block_defer() {
    let src = r#"
var log = ""
do:
    defer log = log + "deferred"
    log = log + "body"
log = log + " after"
let _result = log
"#;
    assert_eq!(run_src(src), Value::Str("bodydeferred after".into()));
}

// `return` inside `do:` exits the enclosing function.
#[test]
fn test_do_block_return_exits_function() {
    let src = r#"
def int compute(int x):
    do:
        if x < 0: return -1
        var doubled = x * 2
    0

let _result = compute(-5)
"#;
    assert_eq!(run_src(src), Value::Int(-1));
}

// Nested `do:` blocks — each has its own scope.
#[test]
fn test_do_block_nested() {
    let src = r#"
let _result = do:
    var a = 1
    var b = do:
        var c = 10
        a + c
    b * 2
"#;
    assert_eq!(run_src(src), Value::Int(22));
}

// `do:` as statement — return value is discarded (no must-use error since it is not a call).
#[test]
fn test_do_block_as_statement() {
    let src = r#"
var x = 0
do:
    x = 5
let _result = x
"#;
    assert_eq!(run_src(src), Value::Int(5));
}

// do-while still works after the parser change.
#[test]
fn test_do_while_still_works() {
    let src = r#"
var i = 0
do:
    i = i + 1
while i < 3
let _result = i
"#;
    assert_eq!(run_src(src), Value::Int(3));
}

// ── {} / {=} literal syntax ──────────────────────────────────────────────────

// {} is an empty SET, not an empty dict.
#[test]
fn test_empty_brace_is_set() {
    let src = r#"
let s = {}
let _len = s.len()
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    // s is a Set, length 0
    assert_eq!(get_var(&interp, "_len"), Value::Int(0));
    assert!(matches!(get_var(&interp, "s"), Value::Set(_)));
}

// {=} is an empty DICT.
#[test]
fn test_empty_dict_literal() {
    let src = r#"
var d = {=}
d.set("x", 42)
let _v = d["x"]
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_v"), Value::Int(42));
    assert!(matches!(get_var(&interp, "d"), Value::Dict(_)));
}

// Non-empty set literals still work.
#[test]
fn test_set_literal_nonempty() {
    let src = r#"
let s = {1, 2, 3}
let _len = s.len()
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_len"), Value::Int(3));
}

// Non-empty dict literals still work.
#[test]
fn test_dict_literal_nonempty() {
    let src = r#"
let d = {"a" = 1, "b" = 2}
let _a = d["a"]
let _b = d["b"]
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_a"), Value::Int(1));
    assert_eq!(get_var(&interp, "_b"), Value::Int(2));
}

// {string=int} type annotation with {=} initialiser.
#[test]
fn test_dict_type_annotation_with_empty_literal() {
    let src = r#"
var {string=int} map = {=}
map.set("hello", 99)
let _v = map["hello"]
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_v"), Value::Int(99));
}

// ── Rust macro calls (name!(...)) ────────────────────────────────────────────

// format!("{}", x) returns a formatted string.
#[test]
fn test_macro_format() {
    let src = r#"
let _s = format!("{} + {} = {}", 1, 2, 3)
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_s"), Value::Str("1 + 2 = 3".to_string()));
}

// format! with named format specifiers.
#[test]
fn test_macro_format_hex() {
    let src = r#"
let _s = format!("{:x}", 255)
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_s"), Value::Str("ff".to_string()));
}

// vec![...] produces a boring Array.
#[test]
fn test_macro_vec() {
    let src = r#"
let v = vec![10, 20, 30]
let _len = v.len()
let _0  = v[0]
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_len"), Value::Int(3));
    assert_eq!(get_var(&interp, "_0"),   Value::Int(10));
}

// println! and print! return Void — no must-use error.
#[test]
fn test_macro_println_void() {
    let src = r#"
println!("hello {}", "world")
print!("x")
"#;
    let (_, res) = run(src);
    res.expect("no error");
}

// assert! passes on true.
#[test]
fn test_macro_assert_ok() {
    let src = r#"
assert!(1 + 1 == 2)
assert!(true, "must be true")
"#;
    let (_, res) = run(src);
    res.expect("no error");
}

// assert! fails on false.
#[test]
fn test_macro_assert_fail() {
    let src = r#"
assert!(1 == 2)
"#;
    let (_, res) = run(src);
    assert!(res.is_err());
}

// assert_eq! passes when equal.
#[test]
fn test_macro_assert_eq_ok() {
    let src = r#"
assert_eq!(2 + 2, 4)
"#;
    let (_, res) = run(src);
    res.expect("no error");
}

// assert_eq! fails when not equal.
#[test]
fn test_macro_assert_eq_fail() {
    let src = r#"
assert_eq!(1, 2)
"#;
    let (_, res) = run(src);
    assert!(res.is_err());
}

// assert_ne! passes when not equal.
#[test]
fn test_macro_assert_ne_ok() {
    let src = r#"
assert_ne!(1, 2)
"#;
    let (_, res) = run(src);
    res.expect("no error");
}

// panic! produces a runtime error.
#[test]
fn test_macro_panic() {
    let src = r#"
panic!("something went wrong")
"#;
    let (_, res) = run(src);
    assert!(res.is_err());
}

// todo! produces a runtime error.
#[test]
fn test_macro_todo() {
    let src = r#"
todo!()
"#;
    let (_, res) = run(src);
    assert!(res.is_err());
}

// dbg! returns the value (can be assigned).
#[test]
fn test_macro_dbg() {
    let src = r#"
let _v = dbg!(42)
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_v"), Value::Int(42));
}

// concat! joins values into a string.
#[test]
fn test_macro_concat() {
    let src = r#"
let _s = concat!("hello", " ", "world")
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_s"), Value::Str("hello world".to_string()));
}

// Bracket syntax: vec![...] — already tested above, but also check with square syntax.
#[test]
fn test_macro_bracket_syntax() {
    let src = r#"
let v = vec![1, 2, 3]
let _len = v.len()
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_len"), Value::Int(3));
}

// Macro call as expression inside let.
#[test]
fn test_macro_in_expression() {
    let src = r#"
let n = 7
let _s = format!("n = {}", n)
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_s"), Value::Str("n = 7".to_string()));
}

// ── Native Rust types (no import needed) ────────────────────────────────────

// HashMap, HashSet, Vec, String are pre-registered — no `use` required.
#[test]
fn test_native_hashmap_no_import() {
    let src = r#"
var m = HashMap()
m.set("x", 10)
let _v = m["x"]
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_v"), Value::Int(10));
    assert!(matches!(get_var(&interp, "m"), Value::Dict(_)));
}

#[test]
fn test_native_hashset_no_import() {
    let src = r#"
var s = HashSet()
let _len = s.len()
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_len"), Value::Int(0));
    assert!(matches!(get_var(&interp, "s"), Value::Set(_)));
}

#[test]
fn test_native_vec_no_import() {
    let src = r#"
var v = Vec()
v.push(1)
v.push(2)
let _len = v.len()
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_len"), Value::Int(2));
    assert!(matches!(get_var(&interp, "v"), Value::Array(_)));
}

#[test]
fn test_native_btreemap_no_import() {
    let src = r#"
var m = BTreeMap()
m.set("a", 1)
let _v = m["a"]
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_v"), Value::Int(1));
}

#[test]
fn test_native_string_constructor_no_import() {
    let src = r#"
let s = String()
let _len = s.len()
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_len"), Value::Int(0));
}

// Explicit import still works (no-op since HashMap is already native).
#[test]
fn test_rust_hashmap_constructor() {
    let src = r#"
use std.collections.HashMap
var m = HashMap()
m.set("x", 10)
let _v = m["x"]
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_v"), Value::Int(10));
    assert!(matches!(get_var(&interp, "m"), Value::Dict(_)));
}

// HashMap.new() is also supported for Rust-fluent users.
#[test]
fn test_rust_hashmap_new_method() {
    let src = r#"
use std.collections.HashMap
var m = HashMap.new()
m.set("k", 99)
let _v = m["k"]
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_v"), Value::Int(99));
}

// HashSet() → empty Set, equivalent to {}.
#[test]
fn test_rust_hashset_constructor() {
    let src = r#"
use std.collections.HashSet
var s = HashSet()
let _len = s.len()
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_len"), Value::Int(0));
    assert!(matches!(get_var(&interp, "s"), Value::Set(_)));
}

// Vec() → empty Array, equivalent to [].
#[test]
fn test_rust_vec_constructor() {
    let src = r#"
use std.collections.Vec
var v = Vec()
v.push(1)
v.push(2)
let _len = v.len()
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_len"), Value::Int(2));
    assert!(matches!(get_var(&interp, "v"), Value::Array(_)));
}

// Unknown Rust types (e.g. File, BufReader) produce an opaque Object.
#[test]
fn test_rust_opaque_type() {
    let src = r#"
use std.fs.File
let f = File()
let _name = "ok"
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    match get_var(&interp, "f") {
        Value::Object(inner) => assert_eq!(inner.borrow().type_name, "File"),
        other => panic!("expected Object, got {:?}", other),
    }
}

// Multiple items from the same std module.
#[test]
fn test_rust_multi_import() {
    let src = r#"
use std.collections.HashMap, HashSet, Vec
var m = HashMap()
var s = HashSet()
var v = Vec()
m.set("a", 1)
let _m = m["a"]
let _slen = s.len()
let _vlen = v.len()
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_m"), Value::Int(1));
    assert_eq!(get_var(&interp, "_slen"), Value::Int(0));
    assert_eq!(get_var(&interp, "_vlen"), Value::Int(0));
}

// ─── Associated types ─────────────────────────────────────────────────────────

#[test]
fn test_assoc_type_basic() {
    let src = r#"
trait Producible:
    type Output
    req Output produce()

struct IntProducer as Producible:
    type Output = int
    req Output produce(): 42

let p = IntProducer()
let _result = p.produce()
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(42));
}

#[test]
fn test_assoc_type_self_prefix() {
    // Self.Output and bare Output are equivalent in method signatures
    let src = r#"
trait Wrapper:
    type Inner
    req Self.Inner unwrap()
    req Inner peek()

struct BoxInt as Wrapper:
    type Inner = int
    req Self.Inner unwrap(): 99
    req Inner peek(): 99

let b = BoxInt()
let _result = b.unwrap()
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(99));
}

#[test]
fn test_assoc_type_constraint() {
    // `type Display as string` — constraint, concrete type must be string
    let src = r#"
trait Displayable:
    type Display as string
    req Display show()

struct Point as Displayable:
    type Display = string
    req Display show(): "point"

let p = Point()
let _result = p.show()
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Str("point".into()));
}

#[test]
fn test_assoc_type_used_in_param() {
    // Associated type used as parameter type too
    let src = r#"
trait Transformer:
    type Input
    type Output
    req Output transform(Input value)

struct Doubler as Transformer:
    type Input = int
    type Output = int
    req Output transform(Input value): value * 2

let d = Doubler()
let _result = d.transform(21)
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Int(42));
}

// {string=int} type + HashMap() constructor — the canonical boring dict idiom.
#[test]
fn test_hashmap_as_dict_type() {
    let src = r#"
use std.collections.HashMap
var {string=int} scores = HashMap()
scores.set("alice", 42)
scores.set("bob", 17)
let _a = scores["alice"]
let _b = scores["bob"]
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_a"), Value::Int(42));
    assert_eq!(get_var(&interp, "_b"), Value::Int(17));
}

// String() constructor.
#[test]
fn test_rust_string_constructor() {
    let src = r#"
use std.string.String
let s = String()
let _len = s.len()
"#;
    let (interp, res) = run(src);
    res.expect("no error");
    assert_eq!(get_var(&interp, "_len"), Value::Int(0));
    assert!(matches!(get_var(&interp, "s"), Value::Str(_)));
}

// ─── loop as expression (break with value) ───────────────────────────────────

#[test]
fn test_loop_expr_break_value() {
    let src = r#"
let _result = loop:
    break 42
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_loop_expr_conditional_break() {
    let src = r#"
var i = 0
let _result = loop:
    i = i + 1
    if i == 5:
        break i
"#;
    assert_eq!(run_src(src), Value::Int(5));
}

#[test]
fn test_loop_expr_string_break() {
    let src = r#"
var n = 0
let _result = loop:
    n = n + 1
    if n >= 3:
        break "done"
"#;
    assert_eq!(run_src(src), Value::Str("done".into()));
}

#[test]
fn test_loop_stmt_plain_break_unaffected() {
    // Plain `break` (no value) in a loop statement still works
    let src = r#"
var i = 0
loop:
    i = i + 1
    if i == 3:
        break
let _result = i
"#;
    assert_eq!(run_src(src), Value::Int(3));
}

// ─── Trait default implementations ───────────────────────────────────────────

#[test]
fn test_trait_default_impl_used() {
    // Struct doesn't override the default — uses the trait's body
    let src = r#"
trait Greeter:
    req string greet():
        "hello"

struct Bot as Greeter: pass

let b = Bot()
let _result = b.greet()
"#;
    assert_eq!(run_src(src), Value::Str("hello".into()));
}

#[test]
fn test_trait_default_impl_overridden() {
    // Struct provides its own implementation — default is ignored
    let src = r#"
trait Greeter:
    req string greet():
        "hello"

struct FrenchBot as Greeter:
    req string greet(): "bonjour"

let b = FrenchBot()
let _result = b.greet()
"#;
    assert_eq!(run_src(src), Value::Str("bonjour".into()));
}

#[test]
fn test_trait_default_and_abstract_mix() {
    // One method has a default, the other is abstract (must be implemented)
    let src = r#"
trait Animal:
    req string name()            # abstract — must implement
    req string speak():          # default — optional
        "..."

struct Dog as Animal:
    req string name(): "dog"
    # speak() not overridden — uses default

struct Cat as Animal:
    req string name(): "cat"
    req string speak(): "meow"  # overrides default

let d = Dog()
let c = Cat()
let _d = d.speak()
let _c = c.speak()
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_d"), Value::Str("...".into()));
    assert_eq!(get_var(&interp, "_c"), Value::Str("meow".into()));
}

#[test]
fn test_trait_default_conformance_block() {
    // Default used via header declaration (no method override)
    let src = r#"
trait Printable:
    req string describe():
        "object"

struct Point as Printable:
    int x
    int y

let p = Point(x=1, y=2)
let _result = p.describe()
"#;
    assert_eq!(run_src(src), Value::Str("object".into()));
}

// ─── Struct destructuring in match ───────────────────────────────────────────

#[test]
fn test_match_struct_destructure() {
    let src = r#"
struct Point:
    int x
    int y

def int sum(Point p):
    match p:
        Point(x, y): x + y

let _result = sum(Point(x=3, y=4))
"#;
    assert_eq!(run_src(src), Value::Int(7));
}

#[test]
fn test_match_struct_wildcard() {
    let src = r#"
struct Point:
    int x
    int y

def int get_x(Point p):
    match p:
        Point(x, _): x

let _result = get_x(Point(x=10, y=99))
"#;
    assert_eq!(run_src(src), Value::Int(10));
}

#[test]
fn test_match_struct_type_check() {
    // Bare name with no sub-patterns — type check only
    let src = r#"
struct Circle:
    float radius

struct Square:
    float side

def string shape<S>(S s):
    match s:
        Circle(_): "circle"
        Square(_): "square"
        _: "unknown"

let _result = shape(Circle(radius=1.0))
"#;
    assert_eq!(run_src(src), Value::Str("circle".into()));
}

#[test]
fn test_match_struct_nested() {
    let src = r#"
struct Inner:
    int value

struct Outer:
    Inner inner

def int deep(Outer o):
    match o:
        Outer(Inner(v)): v

let _result = deep(Outer(inner=Inner(value=42)))
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

// ── lazy binding tests ────────────────────────────────────────────────────────

#[test]
fn test_lazy_basic_init() {
    let src = r#"
lazy int x
x ?= 42
let _result = x
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_lazy_idempotent() {
    // Second ?= must be a no-op — value stays 10
    let src = r#"
lazy int x
x ?= 10
x ?= 99
let _result = x
"#;
    assert_eq!(run_src(src), Value::Int(10));
}

#[test]
fn test_lazy_string() {
    let src = r#"
lazy string name
name ?= "Alice"
name ?= "Bob"
let _result = name
"#;
    assert_eq!(run_src(src), Value::Str("Alice".into()));
}

#[test]
fn test_lazy_rhs_expression() {
    let src = r#"
lazy int r
r ?= 3 + 4
let _result = r
"#;
    assert_eq!(run_src(src), Value::Int(7));
}

#[test]
fn test_lazy_assign_error() {
    // Plain `=` on a lazy binding must be a runtime error
    let src = r#"
lazy int x
x = 42
"#;
    let (_, res) = run(src);
    assert!(res.is_err(), "expected runtime error for '=' on lazy");
    let msg = format!("{:?}", res.unwrap_err());
    assert!(msg.contains("lazy"), "error should mention 'lazy': {}", msg);
}

#[test]
fn test_question_eq_nil_coalescing_preserved() {
    // ?= on a regular optional var must still work as nil-coalescing
    let src = r#"
var int? opt = nil
opt ?= 55
let _result = opt
"#;
    assert_eq!(run_src(src), Value::Int(55));
}

#[test]
fn test_question_eq_non_nil_no_op() {
    // ?= on a non-nil var must not overwrite
    let src = r#"
var int? opt = 10
opt ?= 99
let _result = opt
"#;
    assert_eq!(run_src(src), Value::Int(10));
}

// ─── Callable structs ────────────────────────────────────────────────────────

#[test]
fn test_callable_struct_req_with_return_type() {
    let src = r#"
struct Adder:
    int base
    req int ():
        base + 10

let a = Adder(base= 5)
let _result = a()
"#;
    assert_eq!(run_src(src), Value::Int(15));
}

#[test]
fn test_callable_struct_def_mutation() {
    let src = r#"
struct Counter:
    var int value = 0
    def ():
        value += 1

var c = Counter()
c()
c()
c()
let _result = c.value
"#;
    assert_eq!(run_src(src), Value::Int(3));
}

#[test]
fn test_callable_struct_req_no_return() {
    // req () with no declared return type — invocation succeeds, result is Void
    let src = r#"
struct Noop:
    int x
    req ():
        x

let n = Noop(x= 7)
n()
let _result = 42
"#;
    assert_eq!(run_src(src), Value::Int(42));
}

#[test]
fn test_callable_struct_not_callable_error() {
    let src = r#"
struct Plain:
    int x

let p = Plain(x= 1)
p()
"#;
    let (_, res) = run(src);
    assert!(res.is_err(), "expected runtime error for non-callable struct");
    let msg = format!("{:?}", res.unwrap_err());
    assert!(
        msg.contains("not callable") || msg.contains("__call__"),
        "error should mention callable: {}",
        msg
    );
}

// ── mut on scalars ────────────────────────────────────────────────────────────

#[test]
fn test_mut_scalar_inferred_int() {
    // `mut x = 42` with no type annotation — rebindable int
    let src = r#"
mut x = 42
x = 99
let _result = x
"#;
    assert_eq!(run_src(src), Value::Int(99));
}

#[test]
fn test_mut_scalar_explicit_int() {
    // `mut int x = 0` — rebindable with explicit type
    let src = r#"
mut int x = 0
x = 7
let _result = x
"#;
    assert_eq!(run_src(src), Value::Int(7));
}

#[test]
#[allow(clippy::approx_constant)] // 3.14 is a test literal, not a stand-in for PI
fn test_mut_scalar_float() {
    let src = r#"
mut float f = 1.0
f = 3.14
let _result = f
"#;
    assert_eq!(run_src(src), Value::Float64(3.14));
}

#[test]
fn test_mut_scalar_bool() {
    let src = r#"
mut b = true
b = false
let _result = b
"#;
    assert_eq!(run_src(src), Value::Bool(false));
}

#[test]
fn test_mut_scalar_rebind_multiple_times() {
    let src = r#"
mut n = 1
n = 2
n = 3
n = 4
let _result = n
"#;
    assert_eq!(run_src(src), Value::Int(4));
}

#[test]
fn test_mut_scalar_in_function() {
    // mut scalar inside a function body
    let src = r#"
int count_up(int start):
    mut i = start
    i = i + 1
    i = i + 1
    return i

let _result = count_up(10)
"#;
    assert_eq!(run_src(src), Value::Int(12));
}

#[test]
fn test_mut_scalar_toplevel() {
    // mut at top level (Item::Let path)
    let (interp, res) = run(r#"
mut int counter = 0
counter = 42
"#);
    assert!(res.is_ok(), "expected no error: {:?}", res);
    assert_eq!(get_var(&interp, "counter"), Value::Int(42));
}

#[test]
fn test_mut_uint_scalar() {
    let src = r#"
mut uint u = 0
u = 100
let _result = u
"#;
    assert_eq!(run_src(src), Value::Int(100)); // uint is represented as Int in the interpreter
}

// ── Array slice syntax ───────────────────────────────────────────────────────

#[test]
fn test_slice_m_to_n() {
    let src = r#"
let a = [10, 20, 30, 40, 50]
let _result = a[1..3]
"#;
    assert_eq!(run_src(src), Value::Array(vec![Value::Int(20), Value::Int(30)].into()));
}

#[test]
fn test_slice_from_start() {
    let src = r#"
let a = [10, 20, 30, 40, 50]
let _result = a[..3]
"#;
    assert_eq!(run_src(src), Value::Array(vec![Value::Int(10), Value::Int(20), Value::Int(30)].into()));
}

#[test]
fn test_slice_to_end() {
    let src = r#"
let a = [10, 20, 30, 40, 50]
let _result = a[2..]
"#;
    assert_eq!(run_src(src), Value::Array(vec![Value::Int(30), Value::Int(40), Value::Int(50)].into()));
}

#[test]
fn test_slice_full() {
    let src = r#"
let a = [1, 2, 3]
let _result = a[..]
"#;
    assert_eq!(run_src(src), Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)].into()));
}

#[test]
fn test_slice_inclusive() {
    let src = r#"
let a = [10, 20, 30, 40, 50]
let _result = a[1..=3]
"#;
    assert_eq!(run_src(src), Value::Array(vec![Value::Int(20), Value::Int(30), Value::Int(40)].into()));
}

#[test]
fn test_slice_empty_range() {
    let src = r#"
let a = [10, 20, 30]
let _result = a[2..1]
"#;
    assert_eq!(run_src(src), Value::Array(vec![].into()));
}

#[test]
fn test_slice_out_of_bounds_clamps() {
    let src = r#"
let a = [1, 2, 3]
let _result = a[1..100]
"#;
    assert_eq!(run_src(src), Value::Array(vec![Value::Int(2), Value::Int(3)].into()));
}

#[test]
fn test_slice_with_var_bounds() {
    let src = r#"
let a = [1, 2, 3, 4, 5]
let lo = 1
let hi = 4
let _result = a[lo..hi]
"#;
    assert_eq!(run_src(src), Value::Array(vec![Value::Int(2), Value::Int(3), Value::Int(4)].into()));
}

#[test]
fn test_slice_in_for_loop() {
    let src = r#"
let a = [10, 20, 30, 40, 50]
var sum = 0
for v in a[1..4]:
    sum += v
let _result = sum
"#;
    assert_eq!(run_src(src), Value::Int(90)); // 20+30+40
}

// ── Nested arrays (array of arrays) ──────────────────────────────────────────

#[test]
fn test_nested_array_literal() {
    let src = r#"
let m = [[1.0, 2.0], [3.0, 4.0]]
let _result = m
"#;
    assert_eq!(
        run_src(src),
        Value::Array(vec![
            Value::Array(vec![Value::Float64(1.0), Value::Float64(2.0)].into()),
            Value::Array(vec![Value::Float64(3.0), Value::Float64(4.0)].into()),
        ].into())
    );
}

#[test]
fn test_nested_array_double_index() {
    let src = r#"
let m = [[1.0, 2.0], [3.0, 4.0]]
let _result = m[0][1]
"#;
    assert_eq!(run_src(src), Value::Float64(2.0));
}

#[test]
fn test_nested_array_comprehension() {
    let src = r#"
let rows = [[float(i * j) for j in 0..4] for i in 0..3]
let _result = rows
"#;
    assert_eq!(
        run_src(src),
        Value::Array(vec![
            Value::Array(vec![Value::Float64(0.0), Value::Float64(0.0), Value::Float64(0.0), Value::Float64(0.0)].into()),
            Value::Array(vec![Value::Float64(0.0), Value::Float64(1.0), Value::Float64(2.0), Value::Float64(3.0)].into()),
            Value::Array(vec![Value::Float64(0.0), Value::Float64(2.0), Value::Float64(4.0), Value::Float64(6.0)].into()),
        ].into())
    );
}

// ── BPE tokenizer viability ────────────────────────────────────────────────

#[test]
fn test_bpe_core_merge_loop() {
    let src = r#"
int findBestMerge([string] tokens, [[string]] merges):
    var int best_rank = -1
    var int best_pos  = -1
    let n  = len(tokens)
    let n1 = n - 1
    for i in 0..n1:
        for rank in 0..len(merges):
            if merges[rank][0] == tokens[i] and merges[rank][1] == tokens[i+1]:
                if best_rank == -1 or rank < best_rank:
                    best_rank = rank
                    best_pos  = i
    best_pos

[string] applyMerge([string] tokens, int pos):
    let merged = tokens[pos] + tokens[pos+1]
    var [string] result = []
    let n = len(tokens)
    var int i = 0
    while i < n:
        if i == pos:
            result.push(merged)
            i += 2
        else:
            result.push(tokens[i])
            i += 1
    result

[string] bpeTokenize(string word, [[string]] merges):
    var tokens = word.chars()
    while true:
        let pos = findBestMerge(tokens, merges)
        if pos == -1:
            break
        tokens = applyMerge(tokens, pos)
    tokens

let vocab   = {"a"=0, "b"=1, "ab"=2, "c"=3, "abc"=4, "d"=5, "cd"=6, "abcd"=7}
let merges  = [["a","b"], ["ab","c"], ["c","d"], ["abc","d"]]

let toks = bpeTokenize("abcd", merges)
var [int] ids = []
for tok in toks:
    ids.push(vocab[tok])
let _result = ids
"#;
    let result = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || format!("{:?}", run_src(src)))
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(result, format!("{:?}", Value::Array(vec![Value::Int(7)].into())));
}

#[test]
fn test_bpe_sentence_encode() {
    let src = r#"
let vocab   = {"h"=0,"e"=1,"l"=2,"o"=3,"he"=4,"hel"=8,"hell"=11,"hello"=13,
               "w"=14,"r"=15,"d"=16,"wo"=17,"wor"=21,"worl"=22,"world"=23," "=24}
let merges  = [["h","e"],["he","l"],["hel","l"],["hell","o"],
               ["w","o"],["wo","r"],["wor","l"],["worl","d"]]

int findBestMerge([string] tokens, [[string]] merges):
    var int best_rank = -1
    var int best_pos  = -1
    let n  = len(tokens)
    let n1 = n - 1
    for i in 0..n1:
        for rank in 0..len(merges):
            if merges[rank][0] == tokens[i] and merges[rank][1] == tokens[i+1]:
                if best_rank == -1 or rank < best_rank:
                    best_rank = rank
                    best_pos  = i
    best_pos

[string] applyMerge([string] tokens, int pos):
    let merged = tokens[pos] + tokens[pos+1]
    var [string] result = []
    let n = len(tokens)
    var int i = 0
    while i < n:
        if i == pos:
            result.push(merged)
            i += 2
        else:
            result.push(tokens[i])
            i += 1
    result

[int] bpeEncode(string word, [[string]] merges, {string=int} vocab):
    var tokens = word.chars()
    while true:
        let pos = findBestMerge(tokens, merges)
        if pos == -1:
            break
        tokens = applyMerge(tokens, pos)
    var [int] ids = []
    for tok in tokens:
        if vocab.contains(tok):
            ids.push(vocab[tok])
        else:
            ids.push(-1)
    ids

var [int] all_ids = []
all_ids.push(bpeEncode("hello", merges, vocab)[0])
all_ids.push(vocab[" "])
all_ids.push(bpeEncode("world", merges, vocab)[0])
let _result = all_ids
"#;
    let result = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || format!("{:?}", run_src(src)))
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(
        result,
        format!("{:?}", Value::Array(vec![Value::Int(13), Value::Int(24), Value::Int(23)].into()))
    );
}

// ── Gap fixes ─────────────────────────────────────────────────────────────────

#[test]
fn test_scientific_notation_float() {
    // 1e-10 should lex as Float, not (Ident "1e") - Int(10)
    assert_eq!(run_src("let _result = 1e-10"), Value::Float64(1e-10));
    assert_eq!(run_src("let _result = 2.5e3"), Value::Float64(2500.0));
    assert_eq!(run_src("let _result = 1e10"), Value::Float64(1e10));
}

#[test]
fn test_sort_by_closure() {
    let src = r#"
struct Item:
    string name
    float score

var items = [Item("c", 3.0), Item("a", 1.0), Item("b", 2.0)]
items.sortBy (x): x.score
let _result = [items[0].name, items[1].name, items[2].name]
"#;
    assert_eq!(
        run_src(src),
        Value::Array(vec![
            Value::Str("a".into()),
            Value::Str("b".into()),
            Value::Str("c".into()),
        ].into())
    );
}

#[test]
fn test_sort_by_descending() {
    let src = r#"
struct Item:
    string name
    float score

var items = [Item("c", 3.0), Item("a", 1.0), Item("b", 2.0)]
items.sortBy (x): -x.score
let _result = [items[0].name, items[1].name, items[2].name]
"#;
    assert_eq!(
        run_src(src),
        Value::Array(vec![
            Value::Str("c".into()),
            Value::Str("b".into()),
            Value::Str("a".into()),
        ].into())
    );
}

// ─── `with` scoped-access blocks (docs/scoped-access-blocks.md) ────────────────

#[test]
fn test_with_actor_write() {
    let src = r#"
struct Counter:
    var int value = 0

    def increment():
        value += 1

var mut c'actor = Counter(0)
with c:
    c.increment()
    c.increment()
    c.increment()
let _result = c.value
"#;
    assert_eq!(run_src(src), Value::Int(3));
}

#[test]
fn test_with_guard_read_only_no_write_back_needed() {
    let src = r#"
struct Cell:
    var int value = 0

    req int peek():
        value

var b'guard = Cell(10)
var int seen = 0
with b:
    seen = b.peek()
let _result = seen
"#;
    assert_eq!(run_src(src), Value::Int(10));
}

#[test]
fn test_with_guard_write_then_read_reflects_mutation() {
    let src = r#"
struct Cell:
    var int value = 0

    def bump():
        value += 1

var mut b'guard = Cell(10)
with b:
    b.bump()
let _result = b.value
"#;
    assert_eq!(run_src(src), Value::Int(11));
}

#[test]
fn test_with_multiple_names() {
    let src = r#"
struct Counter:
    var int value = 0

    def increment():
        value += 1

var mut a'actor = Counter(0)
var mut b'actor = Counter(10)
with a, b:
    a.increment()
    b.increment()
let _result = a.value + b.value
"#;
    assert_eq!(run_src(src), Value::Int(12));
}

#[test]
fn test_with_plain_value_is_noop_passthrough() {
    // A `with` on a value with no GPU/actor/guard qualifier degrades to running the
    // body directly — there is nothing to acquire or write back.
    let src = r#"
var int x = 1
with x:
    x = x + 41
let _result = x
"#;
    assert_eq!(run_src(src), Value::Int(42));
}
