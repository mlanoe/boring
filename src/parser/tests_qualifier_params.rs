//! Qualifier parameters (docs/design-notes/qualifier-generic-params.md): declaration forms,
//! `label'qualifier` arguments in every position, and the `T'elem` use.
use crate::ast::*;

const PRELUDE: &str = "struct Point:\n    var int x = 0\n\n\
    struct Board<'elem>:\n    var [mut Point'elem] cells = []\n\n\
    struct Cell<'q as 'new>:\n    var Point'q pt = Point()\n\n\
    struct Cache<T, 'q as 'inline|owned>:\n    var T key\n    var Point'q pt = Point()\n\n\
    struct Wide<'elem as 'actor|'actor'observed|'inline'observed|'shared'weak|'actor'task>:\n    var Point'elem pt = Point()\n\n";

fn parse(src: &str) -> Result<Program, crate::parser::ParseError> {
    crate::parser::parse(crate::lexer::lex(src).expect("lex"))
}

fn struct_named<'a>(p: &'a Program, name: &str) -> &'a StructDecl {
    p.items.iter().find_map(|i| match i { Item::Struct(s) if s.name == name => Some(s), _ => None }).expect("struct")
}

#[test]
fn declaration_forms_are_encoded_as_qualifier_parameters() {
    let p = parse(&format!("{PRELUDE}def main():\n    print \"x\"\n")).unwrap();
    assert_eq!(struct_named(&p, "Board").type_params, ["^elem"]);
    assert_eq!(struct_named(&p, "Cell").type_params, ["^q:owned,shared,actor,guard"]);
    assert_eq!(struct_named(&p, "Cache").type_params, ["T", "^q:inline,owned"]);
    // The bound of an unbounded parameter is every qualifier a bare `T` may take.
    let (name, bound) = qual_param_decl("^elem").unwrap();
    assert_eq!((name, bound), ("elem", OwnerQual::BARE_MEMBERS.iter().map(|q| vec![q.clone()]).collect::<Vec<_>>()));
    assert_eq!(qual_param_default("^elem"), None);
    // `T'elem` in a field is the parameter, not a lifetime or a named qualifier.
    let cells = &struct_named(&p, "Board").fields[0].ty;
    assert!(format!("{cells:?}").contains("Param(\"elem\")"), "{cells:?}");
}

#[test]
fn bounds_may_use_the_named_groups() {
    let p = parse("struct A<'q as 'one>:\n    int x\n\nstruct B<'q as 'many>:\n    int x\n\ndef main():\n    print \"x\"\n").unwrap();
    assert_eq!(struct_named(&p, "A").type_params, ["^q:inline,owned"]);
    assert_eq!(struct_named(&p, "B").type_params, ["^q:shared,actor,guard"]);
}

fn first_let_type_args(stmt_src: &str) -> Vec<Type> {
    let p = parse(&format!("{PRELUDE}def main():\n    {stmt_src}\n")).unwrap_or_else(|e| panic!("{stmt_src}: {}", e.msg()));
    let main = p.items.iter().find_map(|i| match i { Item::Fn(f) if f.name == "main" => Some(f), _ => None }).unwrap();
    let Stmt::Let(l) = &main.body[0] else { panic!("not a let") };
    if let Some(Expr { kind: ExprKind::GenericCall(_, args, _), .. }) = &l.value { return args.clone(); }
    match l.ty.as_ref().map(|t| t.without_mut()) {
        Some(Type::Generic(_, args)) => args.clone(),
        other => panic!("{stmt_src}: no generic arguments: {other:?}"),
    }
}

#[test]
fn qualifier_arguments_are_positional_carriers_in_every_position() {
    let actor = vec![Type::QualArg(vec![OwnerQual::Actor])];
    assert_eq!(first_let_type_args("let b = Board<elem'actor>()"), actor);
    assert_eq!(first_let_type_args("let Board<elem'actor> b = Board<elem'actor>()"), actor);
    assert_eq!(first_let_type_args("let Board<elem'actor> b = nil"), actor);
    // Mixed with a type argument, in either order, labelled or not.
    let mixed = vec![Type::Named("int".into()), Type::QualArg(vec![OwnerQual::Inline])];
    assert_eq!(first_let_type_args("let c = Cache<int, q'inline>(1)"), mixed);
    assert_eq!(first_let_type_args("let c = Cache<q'inline, T = int>(1)"), mixed);
    assert_eq!(first_let_type_args("let c = Cache<T = int, q'inline>(1)"), mixed);
}

#[test]
fn a_qualified_name_is_an_ordinary_type_for_a_callee_without_that_parameter() {
    // `Pair<T'actor>`-style arguments keep their meaning when the callee declares no such label.
    let args = first_let_type_args("let b = Cell<q'actor>()");
    assert_eq!(args, vec![Type::QualArg(vec![OwnerQual::Actor])]);
    let p = parse("struct Box<T>:\n    var T v\n\ndef main():\n    let b = Box<T'actor>(1)\n").unwrap();
    let main = p.items.iter().find_map(|i| match i { Item::Fn(f) if f.name == "main" => Some(f), _ => None }).unwrap();
    let Stmt::Let(l) = &main.body[0] else { panic!() };
    let Some(Expr { kind: ExprKind::GenericCall(_, args, _), .. }) = &l.value else { panic!() };
    assert!(matches!(&args[0], Type::Qualified(_, OwnerQual::Actor)), "{args:?}");
}

#[test]
fn ext_headers_declare_or_name_a_qualifier() {
    let p = parse(&format!("{PRELUDE}ext Board<'elem>:\n    req int a():\n        1\n\next Board<elem'actor>:\n    req int b():\n        2\n\ndef main():\n    print \"x\"\n")).unwrap();
    let exts: Vec<&ExtDecl> = p.items.iter().filter_map(|i| match i { Item::Ext(e) => Some(e), _ => None }).collect();
    assert_eq!(exts[0].type_params, ["^elem"]);
    assert!(exts[1].type_params.is_empty());
    assert_eq!(exts[1].type_args, vec![Type::QualArg(vec![OwnerQual::Actor])]);
}

#[test]
fn qualifier_use_is_only_valid_inside_its_declaration() {
    let e = parse("struct Point:\n    var int x = 0\n\nstruct P:\n    var Point'elem x = Point()\n").expect_err("not in scope");
    assert!(e.msg().contains("expected an ownership qualifier"), "{}", e.msg());
}

#[test]
fn defaults_and_chains_are_encoded_in_the_declaration() {
    let src = "struct A<'q = 'inline>:\n    int x\n\n\
        struct B<'q as 'inline|owned = 'owned>:\n    int x\n\n\
        struct C<'q as 'actor|'actor'observed = 'actor'observed, 'r as 'new>:\n    int x\n\n\
        def main():\n    print \"x\"\n";
    let p = parse(src).unwrap();
    assert_eq!(struct_named(&p, "A").type_params, ["^q=inline"]);
    assert_eq!(struct_named(&p, "B").type_params, ["^q:inline,owned=owned"]);
    assert_eq!(struct_named(&p, "C").type_params, ["^q:actor,actor'observed=actor'observed", "^r:owned,shared,actor,guard"]);
    let (_, bound) = qual_param_decl("^q:actor,actor'observed=actor'observed").unwrap();
    assert_eq!(bound, vec![vec![OwnerQual::Actor], vec![OwnerQual::Actor, OwnerQual::Observed]]);
    assert_eq!(qual_param_default("^q:actor,actor'observed=actor'observed"), Some(vec![OwnerQual::Actor, OwnerQual::Observed]));
    // `A`'s bound is the full set, so its default is a member.
    assert_eq!(qual_param_default("^q=inline"), Some(vec![OwnerQual::Inline]));
}

#[test]
fn a_chain_argument_carries_every_layer() {
    let chain = vec![Type::QualArg(vec![OwnerQual::Actor, OwnerQual::Observed])];
    assert_eq!(first_let_type_args("let b = Wide<elem'actor'observed>()"), chain);
    assert_eq!(first_let_type_args("let Wide<elem'actor'observed> b = nil"), chain);
    assert_eq!(first_let_type_args("let b = Wide<elem'inline'observed>()"), vec![Type::QualArg(vec![OwnerQual::Inline, OwnerQual::Observed])]);
    assert_eq!(first_let_type_args("let b = Wide<elem'shared'weak>()"), vec![Type::QualArg(vec![OwnerQual::Shared, OwnerQual::Weak])]);
    assert_eq!(first_let_type_args("let b = Wide<elem'actor'task>()"), vec![Type::QualArg(vec![OwnerQual::ActorTask])]);
}

#[test]
fn chain_names_round_trip() {
    for chain in [vec![OwnerQual::Inline], vec![OwnerQual::Actor, OwnerQual::Observed], vec![OwnerQual::ActorTask],
                  vec![OwnerQual::GuardTask, OwnerQual::Observed], vec![OwnerQual::Shared, OwnerQual::Weak]] {
        let name = OwnerQual::chain_name(&chain).unwrap();
        assert_eq!(OwnerQual::chain_from_name(&name), Some(chain.clone()), "{name}");
        assert!(OwnerQual::check_chain(&chain).is_ok(), "{name}");
    }
    assert_eq!(OwnerQual::chain_mangle(&[OwnerQual::Actor, OwnerQual::Observed]).as_deref(), Some("actor_observed"));
    assert!(OwnerQual::check_chain(&[OwnerQual::Shared, OwnerQual::Observed]).is_err());
    assert!(OwnerQual::check_chain(&[OwnerQual::Inline, OwnerQual::Weak]).is_err());
    assert!(OwnerQual::check_chain(&[OwnerQual::Weak]).is_err());
}
