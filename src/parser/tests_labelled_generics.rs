//! Labelled generic arguments (`generic_args`): every position resolves to the same positional
//! arguments as the positional spelling, and misuse is reported with a position.
use crate::ast::*;

const PRELUDE: &str = "struct Pair<K, V>:\n    K a\n    V b\n\nstruct Grid<uint N>:\n    int x\n\n\
    struct Tool:\n    int base\n\n    def T2 conv<T2, U2>(T2 x, U2 y):\n        x\n\n\
    T2 pick<T2, U2>(T2 a, U2 b):\n    a\n\n";

fn parse(src: &str) -> Result<Program, crate::parser::ParseError> {
    crate::parser::parse(crate::lexer::lex(src).expect("lex"))
}

fn main_body(program: &Program) -> &[Stmt] {
    for it in &program.items {
        if let Item::Fn(f) = it { if f.name == "main" { return &f.body; } }
    }
    panic!("no main")
}

/// Debug text of the generic arguments of the first `let` in `main` that has any: the call's
/// `GenericCall` arguments, or the annotation's `Type::Generic` arguments.
fn args_of(stmt_src: &str) -> String {
    let program = parse(&format!("{PRELUDE}def main():\n    {stmt_src}\n")).unwrap_or_else(|e| panic!("{stmt_src}: {}", e.msg()));
    let Stmt::Let(l) = &main_body(&program)[0] else { panic!("not a let") };
    if let Some(Expr { kind: ExprKind::GenericCall(_, args, _), .. }) = &l.value { return format!("{args:?}"); }
    if let Some(Type::Generic(_, args)) = l.ty.as_ref().map(|t| t.without_mut()) { return format!("{args:?}"); }
    panic!("{stmt_src}: no generic arguments found: {l:?}")
}

fn same(labelled: &str, positional: &str) {
    assert_eq!(args_of(labelled), args_of(positional), "{labelled}  vs  {positional}");
}

fn error_of(stmt_src: &str) -> (String, usize, usize) {
    let e = parse(&format!("{PRELUDE}def main():\n    {stmt_src}\n")).expect_err("must be rejected");
    (e.msg(), e.line(), e.col())
}

#[test]
fn labelled_equals_positional_in_a_constructor_call() {
    let pos = "let p = Pair<int, string>(1, \"a\")";
    same("let p = Pair<K = int, V = string>(1, \"a\")", pos);
    same("let p = Pair<V = string, K = int>(1, \"a\")", pos);
    same("let p = Pair<int, V = string>(1, \"a\")", pos);
    same("let p = Pair<K = int, V = string,>(1, \"a\")", pos);
}

#[test]
fn labelled_const_argument() {
    same("let g = Grid<N = 4>(1)", "let g = Grid<4>(1)");
}

#[test]
fn labelled_equals_positional_in_a_type_position() {
    same("let Pair<V = string, K = int> p = Pair<int, string>(1, \"a\")", "let Pair<int, string> p = Pair<int, string>(1, \"a\")");
    same("let Pair<int, V = string> p = Pair<int, string>(1, \"a\")", "let Pair<int, string> p = Pair<int, string>(1, \"a\")");
}

#[test]
fn labelled_equals_positional_for_functions_and_methods() {
    same("let r = pick<U2 = string, T2 = int>(1, \"a\")", "let r = pick<int, string>(1, \"a\")");
    same("let r = Tool.conv<U2 = string, T2 = int>(1, \"a\")", "let r = Tool.conv<int, string>(1, \"a\")");
    same("let r = t.conv<U2 = string, T2 = int>(1, \"a\")", "let r = t.conv<int, string>(1, \"a\")");
    same("let r = t?.conv<U2 = string, T2 = int>(1, \"a\")", "let r = t?.conv<int, string>(1, \"a\")");
}

#[test]
fn labels_are_scoped_per_list() {
    same(
        "let p = Pair<K = int, V = Pair<V = int, K = string>>(1, x)",
        "let p = Pair<int, Pair<string, int>>(1, x)",
    );
}

#[test]
fn ext_header_with_labelled_arguments() {
    let src = format!("{PRELUDE}ext Pair<V = string, K = int>:\n    req int one():\n        1\n");
    let program = parse(&src).expect("parse");
    let ext = program.items.iter().find_map(|i| if let Item::Ext(e) = i { Some(e) } else { None }).expect("ext");
    let pos = parse(&format!("{PRELUDE}ext Pair<int, string>:\n    req int one():\n        1\n")).expect("parse");
    let pext = pos.items.iter().find_map(|i| if let Item::Ext(e) = i { Some(e) } else { None }).expect("ext");
    assert_eq!(format!("{:?}", ext.type_args), format!("{:?}", pext.type_args));
}

#[test]
fn declaration_in_another_file_is_resolved_through_use() {
    let dir = std::env::temp_dir().join(format!("boring_labelled_generics_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    std::fs::write(dir.join("shapes.br"), "pub struct Pair<K, V>:\n    pub K a\n    pub V b\n").expect("lib");
    let main = dir.join("main.br");
    std::fs::write(&main, "use shapes\n\ndef main():\n    let p = Pair<V = string, K = int>(1, \"a\")\n").expect("main");
    crate::infer_actor_params::set_import_context(&main);
    let src = std::fs::read_to_string(&main).expect("read");
    let program = parse(&src).expect("parse");
    let Item::Fn(f) = program.items.iter().find(|i| matches!(i, Item::Fn(_))).expect("fn") else { unreachable!() };
    let Stmt::Let(l) = &f.body[0] else { panic!() };
    let Some(Expr { kind: ExprKind::GenericCall(_, args, _), .. }) = &l.value else { panic!("not a generic call") };
    assert_eq!(format!("{args:?}"), args_of("let p = Pair<int, string>(1, \"a\")"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn errors_carry_the_position_of_the_argument() {
    // `    let p = Pair<` puts the first argument at column 18 (`main` body is line 18).
    let (msg, line, col) = error_of("let p = Pair<K = int, K = string>(1, \"a\")");
    assert!(msg.contains("more than once"), "{msg}");
    assert_eq!((line, col), (18, 27));
    let (msg, _, col) = error_of("let p = Pair<X = int, V = string>(1, \"a\")");
    assert!(msg.contains("unknown generic parameter `X`"), "{msg}");
    assert_eq!(col, 18);
    let (msg, _, col) = error_of("let p = Pair<K = int, string>(1, \"a\")");
    assert!(msg.contains("positional generic argument after a labelled one"), "{msg}");
    assert_eq!(col, 27);
    let (msg, _, col) = error_of("let p = Pair<K = int>(1, \"a\")");
    assert!(msg.contains("missing generic argument `V`"), "{msg}");
    assert_eq!(col, 17);
    let (msg, _, _) = error_of("let p = Pair<int, string, int, K = int>(1, \"a\")");
    assert!(msg.contains("takes 2 generic argument(s)"), "{msg}");
}

#[test]
fn comparisons_are_not_generic_calls() {
    for src in ["let r = a < b", "let r = f(a < b, C == d > (e))"] {
        let program = parse(&format!("def main():\n    {src}\n")).unwrap_or_else(|e| panic!("{src}: {}", e.msg()));
        let Stmt::Let(l) = &main_body(&program)[0] else { panic!() };
        assert!(!matches!(&l.value, Some(Expr { kind: ExprKind::GenericCall(..), .. })), "{src}");
    }
}
