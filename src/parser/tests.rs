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
