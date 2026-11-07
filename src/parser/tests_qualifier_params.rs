//! Qualifier parameters (docs/book.md "Qualifier parameters"): declaration forms,
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

// ── A parameter passed on to a contained struct (`Board<elem'cells>`) ──────────────────

const ENCLOSING: &str = "struct Point:\n    var int x = 0\n\n\
    struct Board<'elem = 'inline>:\n    var [mut Point'elem] cells = []\n\n\
    struct Cell<'q as 'new>:\n    var Point'q pt = Point()\n\n";

#[test]
fn a_passed_parameter_is_a_pass_through_argument() {
    let p = parse(&format!("{ENCLOSING}struct Game<'cells = 'inline>:\n    var Board<elem'cells> board = Board<elem'cells>()\n\ndef main():\n    print \"x\"\n")).unwrap();
    let field = &struct_named(&p, "Game").fields[0];
    assert_eq!(field.ty, Type::Generic("Board".into(), vec![Type::QualArg(vec![OwnerQual::Param("cells".into())])]));
    let Some(Expr { kind: ExprKind::GenericCall(_, args, _), .. }) = &field.default else { panic!("{:?}", field.default) };
    assert_eq!(args, &vec![Type::QualArg(vec![OwnerQual::Param("cells".into())])]);
}

#[test]
fn a_passed_parameter_must_be_declared_and_fit_the_receiving_bound() {
    let msg = |src: &str| parse(&format!("{ENCLOSING}{src}\ndef main():\n    print \"x\"\n")).err().map(|e| e.msg());
    assert!(msg("struct G:\n    var Board<elem'cells> b = Board<elem'inline>()\n").unwrap().contains("unknown qualifier parameter `cells`"));
    assert!(msg("struct G<'c>:\n    var Cell<q'c> b = Cell<q'c>()\n").unwrap().contains("is not within the bound of `q` in `Cell`"));
    assert!(msg("struct G<'c as 'actor|guard>:\n    var Cell<q'c> b = Cell<q'c>()\n").is_none());
    assert!(msg("struct G<'c = 'inline>:\n    var Board<elem'c> b = Board<elem'c>()\n").is_none());
}

// ── Functions and methods (design note, section 12) ─────────────────────────────

fn fn_named<'a>(p: &'a Program, name: &str) -> &'a FnDecl {
    p.items.iter().find_map(|i| match i { Item::Fn(f) if f.name == name => Some(f), _ => None }).expect("fn")
}

const FN_PRELUDE: &str = "struct Point:\n    var int x = 0\n\n\
    struct Board<'elem>:\n    var [mut Point'elem] cells = []\n\n";

#[test]
fn function_qualifier_parameters_are_declared_with_bounds_and_defaults() {
    let p = parse(&format!("{FN_PRELUDE}int a<'q>(Board<elem'q> b):\n    1\n\n\
        def b<'q as 'inline|owned = 'inline>(Board<elem'q> b, Point'q p):\n    pass\n\n\
        def c<T, 'q>(T t, Board<elem'q> b):\n    pass\n\n\
        def main():\n    print \"x\"\n")).unwrap();
    assert_eq!(fn_named(&p, "a").type_params, ["^q"]);
    assert_eq!(fn_named(&p, "b").type_params, ["^q:inline,owned=inline"]);
    assert_eq!(fn_named(&p, "c").type_params, ["T", "^q"]);
    assert!(fn_named(&p, "a").implicit_quals.is_empty());
}

#[test]
fn a_qualifier_written_in_a_signature_is_introduced_by_its_first_occurrence() {
    let p = parse(&format!("{FN_PRELUDE}int one(Board<elem'q> b):\n    1\n\n\
        Board<elem'r> two(Board<elem'q> b, Point'q p):\n    b\n\n\
        def main():\n    print \"x\"\n")).unwrap();
    // One parameter, written once: counted, so the checker can warn about it.
    assert_eq!(fn_named(&p, "one").type_params, ["^q"]);
    assert_eq!(fn_named(&p, "one").implicit_quals, [("q".to_string(), 1)]);
    // Return type first, then the parameters in order of first occurrence; `q` is written twice.
    assert_eq!(fn_named(&p, "two").type_params, ["^r", "^q"]);
    assert_eq!(fn_named(&p, "two").implicit_quals, [("r".to_string(), 1), ("q".to_string(), 2)]);
}

#[test]
fn a_function_parameter_does_not_leak_into_the_next_declaration() {
    // `q` is in scope in `a` only: `Point'q` in `b` introduces its own, `main` has none.
    let p = parse(&format!("{FN_PRELUDE}int a<'q>(Point'q p):\n    1\n\n\
        int b(Point'q p):\n    2\n\n\
        def main():\n    print \"x\"\n")).unwrap();
    assert_eq!(fn_named(&p, "a").type_params, ["^q"]);
    assert_eq!(fn_named(&p, "b").type_params, ["^q"]);
    assert!(fn_named(&p, "main").type_params.is_empty());
    let err = parse(&format!("{FN_PRELUDE}int a<'q>(Point'q p):\n    1\n\nstruct Bad:\n    var Point'q p = Point()\n")).expect_err("error");
    assert!(err.msg().contains("unknown qualifier parameter `q`"), "{}", err.msg());
}

#[test]
fn a_method_declares_its_own_parameter_next_to_the_one_of_its_struct() {
    let p = parse(&format!("{FN_PRELUDE}struct Game<'cells = 'inline>:\n    var mut Board<elem'cells> board = Board<elem'cells>()\n\n    \
        def merge<'q>(Board<elem'q> other, Point'cells p):\n        pass\n\n    \
        def own(Point'cells p):\n        pass\n\n\
        def main():\n    print \"x\"\n")).unwrap();
    let game = struct_named(&p, "Game");
    assert_eq!(game.type_params, ["^cells=inline"]);
    let merge = game.methods.iter().find(|m| m.name == "merge").unwrap();
    // `cells` belongs to the struct, only `q` is the method's.
    assert_eq!(merge.type_params, ["^q"]);
    assert!(game.methods.iter().find(|m| m.name == "own").unwrap().type_params.is_empty());
}

#[test]
fn qualifier_words_are_never_introduced_as_parameters() {
    let p = parse(&format!("{FN_PRELUDE}def f(Point'actor a, Point'owned b, Point'shared c, Point'inline d, Point'guard e, Point'new g, Point'atomic h):\n    pass\n\n\
        def main():\n    print \"x\"\n")).unwrap();
    assert!(fn_named(&p, "f").type_params.is_empty());
}

#[test]
fn a_function_cannot_shadow_a_parameter_of_its_struct_and_a_body_cannot_introduce_one() {
    let err = parse(&format!("{FN_PRELUDE}struct Game<'q>:\n    var Point'q p = Point()\n\n    def m<'q>(Point'q x):\n        pass\n")).expect_err("error");
    assert!(err.msg().contains("qualifier parameter `q` of `m` has the same name as one of the enclosing declaration"), "{}", err.msg());
    let err = parse(&format!("{FN_PRELUDE}def f(int x):\n    let Point'r p = Point()\n")).expect_err("error");
    assert!(err.msg().contains("unknown qualifier parameter `r`"), "{}", err.msg());
}

#[test]
fn trait_methods_do_not_take_qualifier_parameters() {
    let err = parse(&format!("{FN_PRELUDE}trait Sized:\n    req int size<'q>(Board<elem'q> b)\n")).expect_err("error");
    assert!(err.msg().contains("trait method `size` cannot declare a qualifier parameter"), "{}", err.msg());
}

#[test]
fn a_declared_default_or_bound_is_checked_at_the_function() {
    for (sig, fragment) in [
        ("def f<'q as 'inline|owned = 'actor>(Point'q p):", "default `'actor` of qualifier parameter `q` is outside its bound"),
        ("def f<'actor>(int x):", "`'actor` is a qualifier, it cannot name a qualifier parameter"),
        ("def f<'q as 'weak>(Point'q p):", "`'weak` cannot bound a qualifier parameter"),
    ] {
        let err = parse(&format!("{FN_PRELUDE}{sig}\n    pass\n")).expect_err("error");
        assert!(err.msg().contains(fragment), "{sig}: {}", err.msg());
    }
}

// ── Gaps closed late: a suffix on a passed parameter, qualified patterns ─────────────

const HELD: &str = "struct Point:\n    var int x = 0\n\n\
    struct Held<'q as 'actor'observed|'guard'observed>:\n    mut Point'q pt = Point()\n\n\
    enum Slot<'elem = 'inline>:\n    Empty\n    Full(Point'elem)\n\n";

#[test]
fn a_passed_parameter_may_carry_a_suffix() {
    let p = parse(&format!("{HELD}struct Wrap<'c as 'actor|'guard>:\n    var Held<q'c'observed> inner = Held<q'c'observed>()\n\ndef main():\n    print \"x\"\n")).unwrap();
    let field = &struct_named(&p, "Wrap").fields[0];
    let chain = vec![OwnerQual::Param("c".into()), OwnerQual::Observed];
    assert_eq!(field.ty, Type::Generic("Held".into(), vec![Type::QualArg(chain.clone())]));
    assert_eq!(OwnerQual::pass_through(&chain), Some(("c", &[OwnerQual::Observed][..])));
    // The bound is checked member by member, suffix included.
    let msg = parse(&format!("{HELD}struct Bad<'c>:\n    var Held<q'c'observed> inner = Held<q'c'observed>()\n\ndef main():\n    print \"x\"\n")).err().map(|e| e.msg()).unwrap();
    assert!(msg.contains("`'inline'observed` once `'observed` is added is outside it"), "{msg}");
}

#[test]
fn a_qualified_pattern_may_name_the_instance() {
    let src = format!("{HELD}def main():\n    let Slot<elem'inline> s = Slot.Empty\n    match s:\n        Slot<elem'inline>.Full(z): print \"f\"\n        Slot.Empty: print \"e\"\n");
    let p = parse(&src).unwrap();
    let Some(Item::Fn(main)) = p.items.iter().find(|i| matches!(i, Item::Fn(f) if f.name == "main")) else { panic!() };
    let Some(Stmt::Match(m)) = main.body.iter().find(|s| matches!(s, Stmt::Match(_))) else { panic!() };
    // The pass leaves the qualified spelling to `qual_infer`, which names the instance; the first arm
    // keeps its explicit arguments in the name until then.
    let names: Vec<&str> = m.arms.iter().flat_map(|a| a.patterns.iter()).filter_map(|p| match p { Pattern::Variant(n, _) => Some(n.as_str()), _ => None }).collect();
    assert!(names.iter().any(|n| n.ends_with("::Full")), "{names:?}");
    assert!(names.iter().any(|n| n.ends_with("::Empty")), "{names:?}");
}
