// `[mut T]` -> `[mut T'actor]` inference (src/infer_actor_elems.rs) on the special targets:
// a host-only struct in a file with a kernel is promoted on `--target simd`; a struct the kernel
// mentions and every struct under `boring build --linux` (no lock-aware element access) are not.
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const KERNEL: &str = "\
kernel Scale:
    let [float32]'unified data

    init([float32]'unified d):
        data = d

    def ():
        data[.thread.x] = data[.thread.x] * 2.0
";

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("mut_elem_alias_targets").join(name);
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn build(d: &PathBuf, args: &[&str], out: &str) -> String {
    let r = Command::new(env!("CARGO_BIN_EXE_boring")).args(args).arg("main.br").current_dir(d).output().unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    fs::read_to_string(d.join(out)).unwrap()
}

#[test]
fn simd_host_only_struct_is_promoted_in_a_kernel_file() {
    let d = dir("simd");
    fs::write(d.join("main.br"), format!("struct Point:\n    var float x\n\n{KERNEL}\ndef main() throws:\n    mut [mut Point] arr = [Point(1.0)]\n    let q = arr[0]\n    arr[0].x = 9.0\n    print q.x\n    mut d = [1.0, 2.0]\n    mut k = Scale(d)\n    kernel:\n        k(block = 2, grid = 1)\n")).unwrap();
    let rs = build(&d, &["build", "--target", "simd"], "main_simd/src/main.rs");
    assert!(rs.contains("Vec<Arc<std::sync::Mutex<Point>>>"), "{rs}");
}

#[test]
fn kernel_mentioned_struct_stays_plain() {
    let d = dir("kernel_struct");
    fs::write(d.join("main.br"), "struct Point:\n    var float32 x\n\nkernel Scale:\n    let [Point]'unified data\n\n    init([Point]'unified d):\n        data = d\n\n    def ():\n        data[.thread.x].x = 2.0\n\ndef main() throws:\n    mut [mut Point] arr = [Point(1.0)]\n    let q = arr[0]\n    arr[0].x = 9.0\n    print q.x\n").unwrap();
    let r = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--target", "simd", "main.br"]).current_dir(&d).output().unwrap();
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let rs = fs::read_to_string(d.join("main_simd/src/main.rs")).unwrap();
    assert!(!rs.contains("Arc<std::sync::Mutex<Point>>"), "{}\n{rs}", String::from_utf8_lossy(&r.stderr));
}

#[test]
fn linux_target_keeps_plain_elements() {
    let d = dir("linux");
    fs::write(d.join("main.br"), "struct Point:\n    var int x\n\nint fix():\n    mut [mut Point] arr = [Point(1)]\n    let q = arr[0]\n    arr[0].x = 9\n    q.x\n").unwrap();
    let rs = build(&d, &["build", "--linux"], "main_linux/src/lib.rs");
    assert!(rs.contains("Vec<Point>") && !rs.contains("Mutex<Point>"), "{rs}");
}

/// metal/cuda/rocm host code: `arr[i].x = v` must write the element, not a `.clone()` of it.
#[test]
fn gpu_host_field_write_through_index_is_in_place() {
    for target in ["metal", "cuda", "rocm"] {
        let d = dir(&format!("write_{target}"));
        fs::write(d.join("main.br"), format!("struct Point:\n    var float x\n\n{KERNEL}\ndef main() throws:\n    mut [mut Point'inline] arr = [Point(1.0)]\n    arr[0].x = 9.0\n    mut d = [1.0, 2.0]\n    mut k = Scale(d)\n    kernel:\n        k(block = 2, grid = 1)\n")).unwrap();
        let rs = build(&d, &["build", "--target", target], &format!("main_{target}/src/main.rs"));
        assert!(rs.contains("arr[(0) as usize].x = 9.0") && !rs.contains(".clone().x = 9.0"), "{target}:\n{rs}");
    }
}

/// metal/cuda/rocm host code honours the inferred `'actor` elements: handle clone for the alias,
/// locked field write/read, wrapped literal and `push`.
#[test]
fn gpu_host_actor_elements() {
    for target in ["metal", "cuda", "rocm"] {
        let d = dir(&format!("actor_{target}"));
        fs::write(d.join("main.br"), format!("struct Point:\n    var float x\n\n{KERNEL}\ndef main() throws:\n    mut [mut Point] arr = [Point(1.0)]\n    let q = arr[0]\n    arr[0].x = 9.0\n    arr.push(Point(2.0))\n    arr[0] = Point(3.0)\n    print q.x\n    mut d = [1.0, 2.0]\n    mut k = Scale(d)\n    kernel:\n        k(block = 2, grid = 1)\n")).unwrap();
        let rs = build(&d, &["build", "--target", target], &format!("main_{target}/src/main.rs"));
        for want in [
            "Vec<std::sync::Arc<std::sync::Mutex<Point>>> = vec![std::sync::Arc::new(std::sync::Mutex::new(Point",
            "let q = arr[(0) as usize].clone();",
            "arr[(0) as usize].lock().unwrap().x = 9.0;",
            "arr.push(std::sync::Arc::new(std::sync::Mutex::new(Point",
            "arr[(0) as usize] = std::sync::Arc::new(std::sync::Mutex::new(Point",
            "q.lock().unwrap().x.clone()",
        ] {
            assert!(rs.contains(want), "{target}: missing `{want}`:\n{rs}");
        }
    }
}

/// Struct-field `[mut T]` / `{K = mut V}` collections read by the metal/cuda/rocm host emitters.
#[test]
fn gpu_host_actor_struct_field_collections() {
    for target in ["metal", "cuda", "rocm"] {
        let d = dir(&format!("field_{target}"));
        fs::write(d.join("main.br"), format!("struct Point:\n    var float x\n\nstruct Board:\n    var [mut Point] cells = []\n    var {{string = mut Point}} named = {{=}}\n\n{KERNEL}\ndef main() throws:\n    mut b = Board()\n    b.cells.push(Point(5.0))\n    b.named[\"k\"] = Point(1.0)\n    let s = b.cells[0]\n    b.cells[0].x = 50.0\n    print s.x\n    mut d = [1.0, 2.0]\n    mut k = Scale(d)\n    kernel:\n        k(block = 2, grid = 1)\n")).unwrap();
        let rs = build(&d, &["build", "--target", target], &format!("main_{target}/src/main.rs"));
        for want in [
            "b.cells.push(std::sync::Arc::new(std::sync::Mutex::new(Point",
            "std::sync::Arc::<str>::from(\"k\")",
            "let s = b.cells[(0) as usize].clone();",
            "b.cells[(0) as usize].lock().unwrap().x = 50.0;",
        ] {
            assert!(rs.contains(want), "{target}: missing `{want}`:\n{rs}");
        }
    }
}

/// Two structs declare a field of the same name, only one as an `'actor` collection: the host
/// emitters resolve the receiver's struct instead of going by field name.
#[test]
fn gpu_host_same_field_name_in_two_structs() {
    for target in ["metal", "cuda", "rocm"] {
        let d = dir(&format!("conflict_{target}"));
        fs::write(d.join("main.br"), format!("struct Point:\n    var float x\n\nstruct A:\n    var [mut Point] items = []\n\nstruct B:\n    var [mut Point'inline] items = []\n\n{KERNEL}\ndef main() throws:\n    mut a = A()\n    mut b = B()\n    a.items.push(Point(1.0))\n    b.items.push(Point(2.0))\n    let q = a.items[0]\n    a.items[0].x = 5.0\n    b.items[0].x = 6.0\n    print q.x\n    mut d = [1.0, 2.0]\n    mut k = Scale(d)\n    kernel:\n        k(block = 2, grid = 1)\n")).unwrap();
        let rs = build(&d, &["build", "--target", target], &format!("main_{target}/src/main.rs"));
        for want in [
            "a.items[(0) as usize].lock().unwrap().x = 5.0;",
            "b.items[(0) as usize].x = 6.0;",
            "b.items.push(Point { x: 2.0 })",
        ] {
            assert!(rs.contains(want), "{target}: missing `{want}`:\n{rs}");
        }
    }
}
