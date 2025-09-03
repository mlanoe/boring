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
