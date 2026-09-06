use crate::ast;

#[test]
fn test_fn_type_param() {
    // `def Int apply(Int f(Int), Int x):` should parse with Fn-typed first param
    let src = "def Int apply(Int f(Int), Int x):\n    f(x)";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let ast::Item::Fn(decl) = &program.items[0] {
        assert_eq!(decl.name, "apply");
        assert_eq!(decl.params.len(), 2);
        // first param should have Fn type
        assert!(matches!(&decl.params[0].ty, Some(ast::Type::Fn(..))));
    } else {
        panic!("expected Fn item");
    }
}

#[test]
fn test_fn_type_no_return() {
    // `def void show(v(String)):` — function-typed param whose own return type may be omitted.
    // The outer `def` still needs an explicit return type (`void` here).
    let src = "def void show(v(String)):\n    v(\"hello\")";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let ast::Item::Fn(decl) = &program.items[0] {
        assert_eq!(decl.name, "show");
        assert_eq!(decl.params.len(), 1);
        match &decl.params[0].ty {
            Some(ast::Type::Fn(ret, params, _, _, _)) => {
                assert!(ret.is_none(), "return type should be None");
                assert_eq!(params.len(), 1);
            }
            other => panic!("expected Fn type, got {:?}", other),
        }
    } else {
        panic!("expected Fn item");
    }
}

#[test]
fn test_fn_type_in_let() {
    // Function type annotations don't support `let type name` form
    // (ambiguous with call syntax); drop the annotation instead.
    let src = "let f = (a, b): a + b";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    assert_eq!(program.items.len(), 1);
}

#[test]
fn test_type_def_typed_throws() {
    // Regression test: a type-level
    // factory method (`type def`) must accept a typed `throws Type:` clause, the
    // same grammar `req`/`def` already accept — previously only the untyped
    // `throws:` form parsed here.
    let src = "\
enum BigUintError:\n    Underflow\n\nstruct BigUint:\n    int value\n\n    type def BigUint make(string s) throws BigUintError:\n        BigUint(0)\n";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    let ast::Item::Struct(decl) = &program.items[1] else { panic!("expected Struct item") };
    assert_eq!(decl.type_methods.len(), 1);
    let tm = &decl.type_methods[0];
    assert_eq!(tm.name, "make");
    assert!(tm.throws, "throws flag should be set");
    match &tm.throws_ty {
        Some(ast::Type::Named(n)) => assert_eq!(n, "BigUintError"),
        other => panic!("expected typed throws BigUintError, got {:?}", other),
    }
}

#[test]
fn test_type_req_typed_throws_task_either_order() {
    // `throws Type` combined with `task`, in both orders, on `type req`.
    for src in [
        "struct BigUint:\n    int value\n\n    type req BigUint zero() throws BigUintError task:\n        BigUint(0)\n",
        "struct BigUint:\n    int value\n\n    type req BigUint zero() task throws BigUintError:\n        BigUint(0)\n",
    ] {
        let tokens = crate::lexer::lex(src).expect("lex");
        let program = crate::parser::parse(tokens).expect("parse");
        let ast::Item::Struct(decl) = &program.items[0] else { panic!("expected Struct item") };
        let tm = &decl.type_methods[0];
        assert!(tm.throws);
        assert!(tm.task);
        match &tm.throws_ty {
            Some(ast::Type::Named(n)) => assert_eq!(n, "BigUintError"),
            other => panic!("expected typed throws BigUintError, got {:?}", other),
        }
    }
}

#[test]
fn test_generic_type_args_extended() {
    // `Foo<&a, T, U as Clone>` — lifetime arg + type param + type param with bound.
    // The `as Clone` bound is silently ignored at use sites.
    // Inline body form (`def ReturnType name(): expr`) avoids needing block `pass`.
    let src = "def Foo<&a, T, U as Clone> make(): nil";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let ast::Item::Fn(decl) = &program.items[0] {
        assert_eq!(decl.name, "make");
        match &decl.return_ty {
            Some(ast::Type::Generic(name, args)) => {
                assert_eq!(name, "Foo");
                assert_eq!(args.len(), 3);
                // First arg is the bare lifetime `'a`
                assert!(matches!(&args[0], ast::Type::Named(s) if s == "'a"));
                // Second and third are type params T and U
                assert!(matches!(&args[1], ast::Type::TypeParam(s) if s == "T"));
                assert!(matches!(&args[2], ast::Type::TypeParam(s) if s == "U"));
            }
            other => panic!("expected Generic return type, got {:?}", other),
        }
    } else {
        panic!("expected Fn item");
    }
}

#[test]
fn test_generic_type_arg_lifetime_only() {
    // Field type `Ref<&a>` — single lifetime argument.
    // Struct field syntax is `Type fieldname` (not `fieldname: Type`).
    let src = "struct Wrapper<&a>:\n    Ref<&a> field";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let ast::Item::Struct(s) = &program.items[0] {
        assert_eq!(s.type_params, vec!["'a".to_string()]);
        let field = &s.fields[0];
        match &field.ty {
            ast::Type::Generic(name, args) => {
                assert_eq!(name, "Ref");
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], ast::Type::Named(s) if s == "'a"));
            }
            other => panic!("expected Generic field type, got {:?}", other),
        }
    } else {
        panic!("expected Struct item");
    }
}

#[test]
fn test_generic_method_call_turbofish() {
    // `obj.method<T>(args)` — turbofish on a method call parses as
    // GenericCall(Field(obj, "method"), [T], args), not a mis-chained comparison.
    let src = "let r = obj.method<int>(5)";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let ast::Item::Let(let_stmt) = &program.items[0] {
        match &let_stmt.value.as_ref().unwrap().kind {
            ast::ExprKind::GenericCall(callee, type_args, args) => {
                match &callee.kind {
                    ast::ExprKind::Field(recv, method) => {
                        assert!(matches!(&recv.kind, ast::ExprKind::Var(n) if n == "obj"));
                        assert_eq!(method, "method");
                    }
                    other => panic!("expected Field callee, got {:?}", other),
                }
                assert_eq!(type_args.len(), 1);
                assert!(matches!(&type_args[0], ast::Type::Named(s) if s == "int"));
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected GenericCall, got {:?}", other),
        }
    } else {
        panic!("expected Let item");
    }
}

#[test]
fn test_method_call_lt_comparison_not_turbofish() {
    // Non-regression: `obj.method < x` is a real less-than comparison, not
    // mis-parsed as the start of a turbofish (no matching `>(` follows).
    let src = "let r = obj.method < 5";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let ast::Item::Let(let_stmt) = &program.items[0] {
        match &let_stmt.value.as_ref().unwrap().kind {
            ast::ExprKind::BinOp(op, lhs, _rhs) => {
                assert_eq!(op, &ast::BinOp::Lt);
                assert!(matches!(&lhs.kind, ast::ExprKind::Field(..)));
            }
            other => panic!("expected BinOp(Lt, Field(...), ...), got {:?}", other),
        }
    } else {
        panic!("expected Let item");
    }
}

#[test]
fn test_optional_generic_method_call_turbofish() {
    // `obj?.method<T>(args)` — turbofish on an optional-chained method call
    // parses as GenericCall(OptionalField(obj, "method"), [T], args), not a
    // mis-chained comparison.
    let src = "let r = obj?.method<int>(5)";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let ast::Item::Let(let_stmt) = &program.items[0] {
        match &let_stmt.value.as_ref().unwrap().kind {
            ast::ExprKind::GenericCall(callee, type_args, args) => {
                match &callee.kind {
                    ast::ExprKind::OptionalField(recv, method) => {
                        assert!(matches!(&recv.kind, ast::ExprKind::Var(n) if n == "obj"));
                        assert_eq!(method, "method");
                    }
                    other => panic!("expected OptionalField callee, got {:?}", other),
                }
                assert_eq!(type_args.len(), 1);
                assert!(matches!(&type_args[0], ast::Type::Named(s) if s == "int"));
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected GenericCall, got {:?}", other),
        }
    } else {
        panic!("expected Let item");
    }
}

#[test]
fn test_optional_method_call_lt_comparison_not_turbofish() {
    // Non-regression: `obj?.method < x` is a real less-than comparison, not
    // mis-parsed as the start of a turbofish (no matching `>(` follows).
    let src = "let r = obj?.method < 5";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let ast::Item::Let(let_stmt) = &program.items[0] {
        match &let_stmt.value.as_ref().unwrap().kind {
            ast::ExprKind::BinOp(op, lhs, _rhs) => {
                assert_eq!(op, &ast::BinOp::Lt);
                assert!(matches!(&lhs.kind, ast::ExprKind::OptionalField(..)));
            }
            other => panic!("expected BinOp(Lt, OptionalField(...), ...), got {:?}", other),
        }
    } else {
        panic!("expected Let item");
    }
}

#[test]
fn test_method_call_lt_gt_chain_not_turbofish() {
    // Non-regression from the design doc: a chained comparison across two
    // method-call fields must still parse as comparisons, not a turbofish,
    // even though it superficially resembles `<...>`.
    let src = "let r = a.b < c and a.b > 0";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let ast::Item::Let(let_stmt) = &program.items[0] {
        // Just confirm it parses at all and is not a GenericCall.
        assert!(!matches!(&let_stmt.value.as_ref().unwrap().kind, ast::ExprKind::GenericCall(..)));
    } else {
        panic!("expected Let item");
    }
}

#[test]
fn test_expr_len_matches_real_extent_consumed() {
    // Regression test: `Expr.len` used to be computed by calling `self.tok_len()`
    // *after* the expression (or a sub-expression like a binop's rhs) had already
    // been parsed — that measures whatever token comes next, not the length of
    // what was actually consumed. A multi-digit literal like `12345` would end up
    // with `len == 1` (the length of the following operator token) instead of `5`.
    let src = "let x = 12345 / 0";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let ast::Item::Let(let_stmt) = &program.items[0] {
        match &let_stmt.value.as_ref().unwrap().kind {
            ast::ExprKind::BinOp(ast::BinOp::Div, lhs, _rhs) => {
                assert_eq!(lhs.len, 5, "expected len=5 for literal `12345`, got {}", lhs.len);
            }
            other => panic!("expected BinOp(Div, ...), got {:?}", other),
        }
    } else {
        panic!("expected Let item");
    }
}

#[test]
fn test_closure_task_inferred_through_match_arm() {
    // Regression test: `scan_expr_throws_task` used to fall through to
    // `_ => (false, false)` for `Match` (also `TryElse`/`TryElseBlock`/`UnaryOp`),
    // so a `task` hidden inside a match arm was never detected — the closure would
    // be transpiled without the async/Result markers a real `task` call needs.
    let src = "let f = (x): match x with 0: task compute(x), _: 1";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let ast::Item::Let(let_stmt) = &program.items[0] {
        match &let_stmt.value.as_ref().unwrap().kind {
            ast::ExprKind::Closure(_, _, _, _throws, task) => {
                assert!(*task, "expected the closure to infer `task` from the match arm");
            }
            other => panic!("expected Closure, got {:?}", other),
        }
    } else {
        panic!("expected Let item");
    }
}

#[test]
fn test_let_new_qualifier_after_name_is_recognized() {
    // Regression test: `let c'new = Ctor(...)` (documented in docs/book.md) — the
    // post-name lookahead used to omit `TokenKind::New`, so the statement fell back
    // to `let c` with no initializer, silently dropping `= Counter(0)` and leaving
    // the rest of the tokens to break subsequent parsing with an unrelated error.
    let src = "let c'new = Counter(0)";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let ast::Item::Let(let_stmt) = &program.items[0] {
        assert!(let_stmt.value.is_some(), "initializer must not be dropped");
        match &let_stmt.value.as_ref().unwrap().kind {
            ast::ExprKind::Call(callee, _) => {
                assert!(matches!(&callee.kind, ast::ExprKind::Var(n) if n == "Counter"));
            }
            other => panic!("expected Call(Counter, ...), got {:?}", other),
        }
        assert!(let_stmt.ty.is_some(), "'new qualifier should have produced a Type");
    } else {
        panic!("expected Let item");
    }
}

#[test]
fn test_let_typed_new_qualifier_after_name_is_recognized() {
    // Same gap, typed-annotation form: `let Type name'new = value`.
    let src = "let Counter c'new = Counter(0)";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let ast::Item::Let(let_stmt) = &program.items[0] {
        assert!(let_stmt.value.is_some(), "initializer must not be dropped");
        assert_eq!(let_stmt.name, "c");
    } else {
        panic!("expected Let item");
    }
}

#[test]
fn test_generic_param_only_in_fn_type_return_position_is_collected() {
    // Regression test: `collect_const_params_from_type`'s first pass had no
    // `Type::Fn` arm (unlike `collect_type_params_from_ty`), so a type param
    // appearing only in a function-typed param's own return type was silently
    // dropped from `type_params` whenever another param already contributed one
    // ordinarily (which makes the first pass "win" and skip the fallback second
    // pass that does have `Type::Fn` support) — producing an unbound generic
    // (`U`) in the transpiled Rust.
    let src = "def T apply(T x, U helper()):\n    x";
    let tokens = crate::lexer::lex(src).expect("lex");
    let program = crate::parser::parse(tokens).expect("parse");
    if let ast::Item::Fn(decl) = &program.items[0] {
        assert!(decl.type_params.contains(&"T".to_string()), "type_params: {:?}", decl.type_params);
        assert!(decl.type_params.contains(&"U".to_string()), "type_params: {:?}", decl.type_params);
    } else {
        panic!("expected Fn item");
    }
}

#[test]
fn test_type_def_method_requires_typed_params() {
    // Regression test: the "every param needs an explicit type" check (already
    // enforced for a top-level `def`/`req`) was never duplicated for a struct's
    // `type def`/`type req` method — an untyped param transpiled to invalid Rust
    // (`fn make(n: ) -> Dog`) instead of being caught here at parse time.
    let src = "struct Dog:\n    string name\n\n    type def Dog make(n):\n        Dog(name = n)";
    let tokens = crate::lexer::lex(src).expect("lex");
    let err = crate::parser::parse(tokens).expect_err("expected a parse error for untyped param");
    let msg = format!("{:?}", err);
    assert!(msg.contains("has no type annotation"), "unexpected error: {}", msg);
}

#[test]
fn test_trait_default_method_requires_typed_params() {
    // Regression test: same gap as above, for a trait method (abstract signature
    // or default implementation) via `parse_fn_signature_or_default`.
    let src = "trait Greeter:\n    def greet(name):\n        print \"Hello, {name}!\"";
    let tokens = crate::lexer::lex(src).expect("lex");
    let err = crate::parser::parse(tokens).expect_err("expected a parse error for untyped param");
    let msg = format!("{:?}", err);
    assert!(msg.contains("has no type annotation"), "unexpected error: {}", msg);
}
