//! Qualifier parameters (docs/design-notes/qualifier-generic-params.md) on the other targets:
//! the GPU host emitters (`--target metal|cuda|rocm`) reuse the general emission, so a
//! kernel-file program gets the same specialised `Board_actor` / `Board_inline`; `--linux` has no
//! specialisation pass and rejects the feature with one clear message. Text checks only: no
//! GPU, no nvcc/hipcc and no kernel build is needed.
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const PROGRAM: &str = r#"
struct Point:
    var int x = 0

struct Board<'elem>:
    var [mut Point'elem] cells = []

    def add(Point'elem p):
        cells.push(p)

    req int size():
        cells.len()

ext Board<'elem>:
    req int doubled():
        self.size() * 2

enum Slot<'elem as 'one>:
    Empty
    Full(Point'elem)

kernel Scale:
    let [float32]'global src
    mut [float32]'unified out
    init([float32]'global s):
        src = s
        out = [0.0 for ..<4]
    def ():
        let i = .thread.x
        out[i] = src[i] * 2.0

pub req [float32]'kernel'unified scale([float32]'global x) throws:
    mut k = Scale(x)
    k.out

def main():
    mut a = Board<elem'actor>()
    let p = Point(1)
    a.add(p)
    mut b = Board<elem'inline>()
    b.add(Point(2))
    let Slot<elem'inline> s = Slot.Full(Point(3))
    print "{a.size()} {a.doubled()} {b.size()}"
"#;

/// The same instances, none of them written: every qualifier argument is inferred at construction
/// (docs/design-notes/qualifier-generic-params.md, section 8).
const INFERRED_MAIN: &str = r#"def main():
    let Point'actor p = Point(1)
    mut a = Board(cells = [p])
    mut Board<elem'inline> b = Board()
    b.add(Point(2))
    let Point'inline q = Point(3)
    let s = Slot.Full(q)
    print "{a.size()} {a.doubled()} {b.size()}"
"#;

fn inferred_program() -> String {
    PROGRAM.split("def main():").next().unwrap().to_string() + INFERRED_MAIN
}

/// Defaulted and chained parameters (design note, section 10): a bare `Board` is the default
/// instance and `q'actor'observed` is mangled with the whole chain.
const DEFAULTS_PROGRAM: &str = r#"
struct Point:
    var int x = 0

    def bump():
        x += 1

struct Board<'elem = 'inline>:
    var [mut Point'elem] cells = []

    def add(Point'elem p):
        cells.push(p)

    req int size():
        cells.len()

struct Watched<'q as 'actor|'actor'observed = 'actor'observed>:
    mut Point'q pt = Point()

    def bump():
        pt.bump()

kernel Scale:
    let [float32]'global src
    mut [float32]'unified out
    init([float32]'global s):
        src = s
        out = [0.0 for ..<4]
    def ():
        let i = .thread.x
        out[i] = src[i] * 2.0

pub req [float32]'kernel'unified scale([float32]'global x) throws:
    mut k = Scale(x)
    k.out

def main():
    mut a = Board()
    a.add(Point(1))
    mut Board<elem'inline> b = Board()
    b.add(Point(2))
    mut w = Watched()
    w.bump()
    mut w2 = Watched<q'actor>()
    w2.bump()
    print "{a.size()} {b.size()}"
"#;

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("qualifier_params_targets").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn host_rs(target: &str, program: &str, tag: &str) -> String {
    let dir = scratch(&format!("{target}{tag}"));
    let br = dir.join("test.br");
    fs::write(&br, program).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--target", target]).arg(&br).output().unwrap();
    assert!(out.status.success(), "[{target}] build failed:\n{}", String::from_utf8_lossy(&out.stderr));
    fs::read_to_string(dir.join(format!("test_{target}")).join("src").join("main.rs")).unwrap()
}

fn check_host(target: &str) {
    check_host_program(target, PROGRAM, "");
    check_host_program(target, &inferred_program(), "_inferred");
}

fn check_host_program(target: &str, program: &str, tag: &str) {
    let rs = host_rs(target, program, tag);
    for decl in ["struct Board_actor", "struct Board_inline", "enum Slot_inline", "impl Board_actor", "impl Board_inline"] {
        assert!(rs.contains(decl), "[{target}] missing `{decl}`:\n{rs}");
    }
    // The actor instance stores locked elements, the inline one plain ones.
    assert!(rs.contains("cells: /* var */ Vec<Arc<std::sync::Mutex<Point>>>"), "[{target}] {rs}");
    assert!(rs.contains("cells: /* var */ Vec<Point>"), "[{target}] {rs}");
    assert!(rs.contains("fn doubled"), "[{target}] the `ext` block is specialised too:\n{rs}");
    // Nothing unspecialised is left behind.
    for leftover in ["struct Board ", "struct Board<", "Slot::", "compile_error!", "^elem"] {
        assert!(!rs.contains(leftover), "[{target}] leftover `{leftover}`:\n{rs}");
    }
}

fn check_defaults_host(target: &str) {
    let rs = host_rs(target, DEFAULTS_PROGRAM, "_defaults");
    for decl in ["struct Board_inline", "struct Watched_actor_observed", "struct Watched_actor", "impl Board_inline"] {
        assert!(rs.contains(decl), "[{target}] missing `{decl}`:\n{rs}");
    }
    assert!(rs.contains("BoringObserved<Arc<std::sync::Mutex<Point>>>"), "[{target}] {rs}");
    // `Board()` and `Board<elem'inline>()` are one specialisation: a single declaration.
    assert_eq!(rs.matches("struct Board_inline").count(), 1, "[{target}] {rs}");
    for leftover in ["struct Board ", "struct Board<", "struct Watched ", "compile_error!", "^elem"] {
        assert!(!rs.contains(leftover), "[{target}] leftover `{leftover}`:\n{rs}");
    }
}

#[test] fn metal_host_gets_default_and_chain_instances() { check_defaults_host("metal"); }
#[test] fn cuda_host_gets_default_and_chain_instances() { check_defaults_host("cuda"); }
#[test] fn rocm_host_gets_default_and_chain_instances() { check_defaults_host("rocm"); }

#[test]
fn linux_target_rejects_defaulted_and_chained_parameters() {
    let dir = scratch("linux_defaults");
    let br = dir.join("test.br");
    let head = DEFAULTS_PROGRAM.split("kernel Scale:").next().unwrap().to_string();
    for main in ["def main():\n    mut a = Board()\n    print \"x\"\n",
                 "def main():\n    mut w = Watched<q'actor'observed>()\n    print \"x\"\n"] {
        fs::write(&br, format!("{head}{main}")).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--linux"]).arg(&br).output().unwrap();
        assert!(!out.status.success(), "--linux must reject qualifier parameters");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("qualifier parameters") && stderr.contains("not supported on the Rust-for-Linux Linux target"), "{stderr}");
    }
}

#[test] fn metal_host_gets_specialised_instances() { check_host("metal"); }
#[test] fn cuda_host_gets_specialised_instances() { check_host("cuda"); }
#[test] fn rocm_host_gets_specialised_instances() { check_host("rocm"); }

#[test]
fn linux_target_rejects_qualifier_parameters() {
    let dir = scratch("linux");
    let br = dir.join("test.br");
    let head = PROGRAM.split("kernel Scale:").next().unwrap().to_string();
    // Written and inferred instances are both rejected.
    for main in ["def main():\n    mut a = Board<elem'actor>()\n    print \"x\"\n",
                 "def main():\n    let Point'actor p = Point(1)\n    mut a = Board(cells = [p])\n    print \"x\"\n"] {
        fs::write(&br, format!("{head}{main}")).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--linux"]).arg(&br).output().unwrap();
        assert!(!out.status.success(), "--linux must reject qualifier parameters");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("qualifier parameters") && stderr.contains("not supported on the Rust-for-Linux Linux target"), "{stderr}");
    }
}

/// A parameter passed on to a contained struct (design note, "Passing the parameter to an enclosing
/// struct"): `Game_actor` holds `Board_actor`, and the cascade reaches the `ext` block and the enum.
const ENCLOSING_PROGRAM: &str = r#"
struct Point:
    var int x = 0

struct Board<'elem = 'inline>:
    var [mut Point'elem] cells = []

    def add(Point'elem p):
        cells.push(p)

    req int size():
        cells.len()

struct Game<'cells = 'inline>:
    var mut Board<elem'cells> board = Board<elem'cells>()
    var int turn = 0

    def play(Point'cells p):
        board.add(p)
        turn += 1

ext Game<'cells>:
    req int described():
        self.board.size() + self.turn

enum Slot<'cells = 'inline>:
    Empty
    Held(Board<elem'cells>)

kernel Scale:
    let [float32]'global src
    mut [float32]'unified out
    init([float32]'global s):
        src = s
        out = [0.0 for ..<4]
    def ():
        let i = .thread.x
        out[i] = src[i] * 2.0

pub req [float32]'kernel'unified scale([float32]'global x) throws:
    mut k = Scale(x)
    k.out

def main():
    mut Game<cells'actor> g = Game()
    let Point'actor p = Point(1)
    g.play(p)
    mut g2 = Game()
    g2.play(Point(2))
    let Slot<cells'inline> s = Slot.Held(Board())
    print "{g.described()} {g2.described()}"
"#;

fn check_enclosing_host(target: &str) {
    let rs = host_rs(target, ENCLOSING_PROGRAM, "_enclosing");
    for decl in ["struct Game_actor", "struct Game_inline", "struct Board_actor", "struct Board_inline", "enum Slot_inline", "impl Game_actor", "impl Game_inline"] {
        assert!(rs.contains(decl), "[{target}] missing `{decl}`:\n{rs}");
    }
    // `Game_actor` holds `Board_actor`, `Game_inline` holds `Board_inline`.
    let game_actor = rs.split("struct Game_actor").nth(1).unwrap().split('}').next().unwrap();
    assert!(game_actor.contains("Board_actor"), "[{target}] {game_actor}");
    let game_inline = rs.split("struct Game_inline").nth(1).unwrap().split('}').next().unwrap();
    assert!(game_inline.contains("Board_inline"), "[{target}] {game_inline}");
    assert!(rs.contains("fn described"), "[{target}] the `ext` block is specialised too:\n{rs}");
    for leftover in ["struct Game ", "struct Game<", "struct Board ", "struct Board<", "compile_error!", "^cells", "^elem"] {
        assert!(!rs.contains(leftover), "[{target}] leftover `{leftover}`:\n{rs}");
    }
}

#[test] fn metal_host_gets_enclosing_instances() { check_enclosing_host("metal"); }
#[test] fn cuda_host_gets_enclosing_instances() { check_enclosing_host("cuda"); }
#[test] fn rocm_host_gets_enclosing_instances() { check_enclosing_host("rocm"); }

#[test]
fn linux_target_rejects_enclosing_parameters() {
    let dir = scratch("linux_enclosing");
    let br = dir.join("test.br");
    let head = ENCLOSING_PROGRAM.split("kernel Scale:").next().unwrap().to_string();
    for main in ["def main():\n    mut g = Game()\n    print \"x\"\n",
                 "def main():\n    mut Game<cells'actor> g = Game()\n    print \"x\"\n"] {
        fs::write(&br, format!("{head}{main}")).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--linux"]).arg(&br).output().unwrap();
        assert!(!out.status.success(), "--linux must reject qualifier parameters");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("qualifier parameters") && stderr.contains("not supported on the Rust-for-Linux Linux target"), "{stderr}");
    }
}

/// Functions and methods generic over a qualifier (design note, section 12): `count_actor`,
/// `count_inline` and `Game_actor.merge_inline` are plain specialised items on the GPU host
/// targets, and `--linux` rejects the declaration.
const FNS_PROGRAM: &str = r#"
struct Point:
    var int x = 0

struct Board<'elem = 'inline>:
    var [mut Point'elem] cells = []

    req int size():
        cells.len()

int count<'q>(Board<elem'q> b):
    b.size()

int count_twice(Board<elem'q> b):
    count(b) + count(b)

struct Game<'cells = 'inline>:
    var mut Board<elem'cells> board = Board<elem'cells>()
    var int turns = 0

    def merge<'q>(Board<elem'q> other):
        turns += count(other)

kernel Scale:
    let [float32]'global src
    mut [float32]'unified out
    init([float32]'global s):
        src = s
        out = [0.0 for ..<4]
    def ():
        let i = .thread.x
        out[i] = src[i] * 2.0

pub req [float32]'kernel'unified scale([float32]'global x) throws:
    mut k = Scale(x)
    k.out

def main():
    let Board<elem'actor> a = Board()
    let Board<elem'inline> b = Board()
    mut Game<cells'actor> g = Game()
    g.merge(b)
    print "{count(a)} {count_twice(b)} {g.turns}"
"#;

fn check_fns_host(target: &str) {
    let rs = host_rs(target, FNS_PROGRAM, "_fns");
    for decl in ["fn count_actor", "fn count_inline", "fn count_twice_inline", "fn merge_inline", "struct Game_actor", "struct Board_inline"] {
        assert!(rs.contains(decl), "[{target}] missing `{decl}`:\n{rs}");
    }
    // Only the instances that are called exist; nothing generic is left behind.
    assert!(!rs.contains("fn count_owned") && !rs.contains("fn count_twice_actor"), "[{target}] {rs}");
    for leftover in ["fn count(", "fn count<", "fn merge<", "fn merge(", "compile_error!", "^q"] {
        assert!(!rs.contains(leftover), "[{target}] leftover `{leftover}`:\n{rs}");
    }
}

#[test] fn metal_host_gets_specialised_functions() { check_fns_host("metal"); }
#[test] fn cuda_host_gets_specialised_functions() { check_fns_host("cuda"); }
#[test] fn rocm_host_gets_specialised_functions() { check_fns_host("rocm"); }

#[test]
fn linux_target_rejects_qualifier_generic_functions() {
    let dir = scratch("linux_fns");
    let br = dir.join("test.br");
    let head = FNS_PROGRAM.split("kernel Scale:").next().unwrap().to_string();
    // A function, a method and an implicit declaration are all rejected, with or without a call.
    for main in ["def main():\n    let Board<elem'actor> a = Board()\n    print \"{count(a)}\"\n",
                 "def main():\n    print \"x\"\n"] {
        fs::write(&br, format!("{head}{main}")).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--linux"]).arg(&br).output().unwrap();
        assert!(!out.status.success(), "--linux must reject qualifier parameters");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("qualifier parameters") && stderr.contains("not supported on the Rust-for-Linux Linux target"), "{stderr}");
    }
    // A function whose only generic part is an explicit, unused `<'q>`.
    fs::write(&br, "def f<'q>(int x):\n    pass\n\ndef main():\n    print \"x\"\n").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--linux"]).arg(&br).output().unwrap();
    assert!(!out.status.success(), "--linux must reject a declared qualifier parameter");
    assert!(String::from_utf8_lossy(&out.stderr).contains("qualifier parameters"));
}
