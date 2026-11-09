// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// CLI-level edge cases from the test plan of docs/book.md, "Implicit generics: design notes":
// the inputs inference must reject (`run` and `build` agree) rather than guess.
//
// Run with:
//   cargo test --test implicit_generics_errors

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

fn fixture_dir() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("implicit_generics_errors_{}_{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
    std::fs::create_dir_all(&dir).expect("failed to create fixture dir");
    dir
}

fn boring(mode: &[&str], src: &str) -> Output {
    let file = fixture_dir().join("main.br");
    std::fs::write(&file, src).expect("failed to write fixture");
    Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(mode)
        .arg(&file)
        .output()
        .expect("failed to invoke boring")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}


fn both(src: &str) -> [Output; 2] {
    [boring(&["run"], src), boring(&["build", "--emit-rust"], src)]
}

fn expect_error(src: &str, needle: &str) {
    for out in both(src) {
        assert!(!out.status.success(), "expected a failure for:\n{src}");
        assert!(stderr(&out).contains(needle), "wanted `{needle}`, got:\n{}", stderr(&out));
    }
}

#[test]
fn field_typed_only_by_a_later_push_is_an_error() {
    // D3: an empty collection is no evidence, a later `push` does not count.
    expect_error(
        "struct Bag:\n    var items\n\ndef main():\n    mut b = Bag(items = [])\n    b.items.push(3)\n    print b.items.len()\n",
        "cannot infer the type of `Bag.items`",
    );
}

#[test]
fn recursive_function_without_a_return_type_is_an_error() {
    // D4: a return type that depends on a call to itself needs a fixpoint, so it must be written.
    expect_error(
        "def count(int n):\n    if n > 0:\n        return count(n - 1)\n    return 0\n\nprint count(3)\n",
        "`count` is recursive and returns a value",
    );
}

#[test]
fn mixed_instances_in_one_array_are_an_error() {
    // D8: `[Response]` is homogeneous.
    expect_error(
        "struct R:\n    let a\n\ndef x():\n    return R(a = 1)\n\ndef y():\n    return R(a = \"s\")\n\nlet v = [x(), y()]\nprint v.len()\n",
        "",
    );
}

#[test]
fn one_struct_two_instantiations_run_and_build_agree() {
    let src = "struct Resp:\n    let status\n    let payload\n\ndef a():\n    return Resp(status = 1, payload = \"x\")\n\ndef b():\n    return Resp(status = 2, payload = [1, 2])\n\nprint a().payload\nprint b().payload.len()\nprint a().status\n";
    let out = boring(&["run"], src);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(String::from_utf8_lossy(&out.stdout), "x\n2\n1\n");
    assert!(boring(&["build", "--emit-rust"], src).status.success());
}

#[test]
fn show_types_lists_constructions_of_an_imported_struct() {
    let dir = fixture_dir();
    std::fs::write(dir.join("types.br"), "struct Resp:\n    let status\n    let payload\n").unwrap();
    std::fs::write(
        dir.join("main.br"),
        "use types\n\ndef mk():\n    return Resp(status = 200, payload = \"hi\")\n\nprint mk().payload\n",
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["check", "--show-types", "main.br"])
        .current_dir(&dir)
        .output()
        .expect("failed to invoke boring");
    assert!(out.status.success(), "{}", stderr(&out));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("Resp<status = int, payload = string>"), "{text}");
}

#[test]
fn bare_mention_of_an_implicit_struct_in_a_type_is_an_error() {
    // A bare mention used to run under `boring run` and fail in rustc (E0107). Parameters are
    // the exception (phase 5: the function becomes generic), see implicit_generics_struct_params.br.
    let head = "struct Response:\n    status\n    payload\n\n";
    let main = "\ndef main():\n    print \"done\"\n";
    for sig in [
        "Response? make():\n    nil\n",
        "struct Wrap:\n    Response inner\n",
    ] {
        expect_error(&format!("{head}{sig}{main}"), "`Response` has implicit parameters (status, payload)");
    }
    // Written arguments are fine, in both modes.
    let ok = format!("{head}def describe(Response<int, string> r):\n    print \"{{r.status}}\"\n{main}");
    for out in both(&ok) {
        assert!(out.status.success(), "{}", stderr(&out));
    }
}

const COUNTER: &str = "struct Counter:\n    var int n = 0\n\n    def inc():\n        n += 1\n\n";

#[test]
fn method_call_on_an_untyped_parameter_is_specialised_per_typed_call() {
    // Typed specialisation (docs/design-notes/typed-monomorphization-draft.md, phase 1): what
    // `a.inc()` emits depends on the qualifier of the argument, so each typed call gets its own
    // copy of `bump`, in `run` and in `build` alike.
    let src = format!("{COUNTER}def bump(var mut a):\n    a.inc()\n\ndef twice(var mut b):\n    bump(b)\n    bump(b)\n\n\
        def main():\n    var mut Counter'owned o = Counter()\n    var mut Counter'actor c = Counter()\n    twice(o)\n    twice(c)\n    bump(c)\n    print \"{{o.n}} {{c.n}}\"\n");
    let run = boring(&["run"], &src);
    assert!(run.status.success(), "{}", stderr(&run));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "2 3");
    let rust = boring(&["build", "--emit-rust"], &src);
    assert!(rust.status.success(), "{}", stderr(&rust));
    let text = String::from_utf8_lossy(&rust.stdout);
    assert!(text.contains("fn bump_Counter_owned(") && text.contains("(*a).inc();"), "{text}");
    assert!(text.contains("fn bump_Counter_actor(") && text.contains("a.lock().unwrap().inc();"), "{text}");
}

#[test]
fn specialisation_needs_the_type_of_the_argument_to_be_written() {
    let src = format!("{COUNTER}def bump(var mut a):\n    a.inc()\n\ndef main():\n    bump(Counter())\n");
    expect_error(&src, "the type of the argument given to the untyped parameter `a` is not known here");
}

#[test]
fn uncalled_function_with_a_method_call_on_an_untyped_parameter_keeps_its_error() {
    let src = format!("{COUNTER}def bump(var mut a):\n    a.inc()\n\ndef main():\n    print \"hi\"\n");
    expect_error(&src, "cannot use the untyped parameter `a` with the method call `.inc()`");
}

#[test]
fn specialisation_refuses_an_argument_bound_twice_with_different_types() {
    // A shadowing `let` makes the type of `x` at the call ambiguous: reported, never guessed.
    let src = format!("{COUNTER}def bump(var mut a):\n    a.inc()\n\ndef main():\n    var mut Counter'owned x = Counter()\n    var mut Counter'actor x = Counter()\n    bump(x)\n");
    expect_error(&src, "`x` is bound more than once in this function with different (or unwritten) types");
}

#[test]
fn uncalled_function_with_a_field_read_on_an_untyped_parameter_keeps_its_error() {
    let src = format!("{COUNTER}int peek(a):\n    a.n\n\ndef main():\n    print \"hi\"\n");
    expect_error(&src, "cannot use the untyped parameter `a` with the field access `.n`");
}

#[test]
fn operators_between_two_untyped_parameters_are_still_an_error() {
    // Only uses whose meaning depends on the argument's type are specialised.
    expect_error("int add(a, b):\n    a + b\n\ndef main():\n    print \"hi\"\n", "cannot use the untyped parameter `a` with the `+` operator with `b`");
}

#[test]
fn qualifier_slot_on_an_untyped_parameter_refuses_another_written_qualifier() {
    // `a'owned`: the qualifier written after the name of an untyped parameter constrains the argument.
    let src = format!("{COUNTER}def bump(var mut a'owned):\n    a.inc()\n\ndef main():\n    var mut Counter'actor c = Counter()\n    bump(c)\n");
    expect_error(&src, "the parameter `a'owned` of `bump` expects that qualifier, but this argument is written `'actor`");
}

#[test]
fn qualifier_slot_on_an_explicit_generic_refuses_another_written_qualifier() {
    let src = "struct Counter:\n    var int n = 0\n\ndef show<T>(T'owned a):\n    print \"x\"\n\ndef main():\n    var mut Counter'actor c = Counter()\n    show(c)\n";
    expect_error(src, "the parameter `a` of `show` is declared `'owned`-qualified, but `c` is written `'actor`");
}

#[test]
fn qualifier_slot_with_the_matching_qualifier_is_specialised_and_an_uncalled_one_is_kept() {
    let ok = format!("{COUNTER}def bump(var mut a'actor):\n    a.inc()\n\ndef unused(a'owned):\n    print \"{{a}}\"\n\ndef main():\n    var mut Counter'actor c = Counter()\n    bump(c)\n    print \"{{c.n}}\"\n");
    let run = boring(&["run"], &ok);
    assert!(run.status.success(), "{}", stderr(&run));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "1");
    let rust = boring(&["build", "--emit-rust"], &ok);
    assert!(rust.status.success(), "{}", stderr(&rust));
    assert!(String::from_utf8_lossy(&rust.stdout).contains("fn bump_Counter_actor("));
}

#[test]
fn qualified_generic_sugar_is_refused_and_a_plain_type_parameter_carries_its_qualifier() {
    // `<T'q>` is not a declaration form (a plain `T` carries the qualifier of its argument).
    let sugar = "struct Entry<T'q>:\n    let T'q res\n\ndef main():\n    print \"x\"\n";
    expect_error(sugar, "`<T'q>` is not a declaration form");
    // `<T, 'q>` declares a qualifier parameter of its own and is accepted.
    let long = "struct Entry<T, 'q>:\n    let T'q res\n\ndef main():\n    print \"x\"\n";
    for out in both(long) {
        assert!(out.status.success(), "{}", stderr(&out));
    }
}

#[test]
fn untyped_parameter_function_the_pass_does_not_handle_keeps_its_error() {
    // Specialisation covers top-level functions with a unique name of one file. A `mod` function and an
    // overloaded name keep the "write its type" error, in `run` and `build` alike (never an opaque rustc error).
    let in_mod = format!("{COUNTER}mod tools:\n    pub def bump(var mut a):\n        a.inc()\n\ndef main():\n    print \"hi\"\n");
    expect_error(&in_mod, "cannot use the untyped parameter `a` with the method call `.inc()`");
    let overloaded = format!("{COUNTER}def bump(var mut a):\n    a.inc()\n\ndef bump(var mut a, int n):\n    a.inc()\n\ndef main():\n    var mut Counter'actor c = Counter()\n    bump(c)\n");
    expect_error(&overloaded, "cannot use the untyped parameter `a` with the method call `.inc()`");
}
