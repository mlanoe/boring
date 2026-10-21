//! Scalar tile fallback shared by device emitters. Public execution remains
//! gated by the checker until launch and interpreter integration is complete.
use crate::{
    ast::*,
    checker::tensor::{resolve_fields, TileOperation},
};

#[derive(Clone, Copy)]
pub(super) enum Dialect {
    Cuda,
    /// CUDA with opt-in WMMA TF32 (m16n16k8) for eligible tiles.
    CudaTf32,
    Rocm,
    Metal,
    MetalNative,
    Wgsl,
}

/// Render expressions with the owning backend so field renames are preserved.
pub(super) fn emit(
    call: &Expr,
    fields: &[KernelFieldDecl],
    dialect: Dialect,
    mut expression: impl FnMut(&Expr) -> String,
) -> String {
    let op = resolve_fields(call, fields).expect("tensor lowering requires a validated call");
    let row = expression(&op.row);
    let col = expression(&op.col);
    let buffers = op.operands.each_ref().map(|name| {
        expression(&Expr {
            kind: ExprKind::Var(name.clone()),
            line: call.line,
            col: call.col,
            len: call.len,
        })
    });
    if matches!(dialect, Dialect::CudaTf32) {
        return cuda_tf32_source(&op, &buffers, &row, &col);
    }
    if matches!(dialect, Dialect::MetalNative)
        && op.rows == 8
        && op.cols == 8
        && op.k.is_multiple_of(8)
    {
        metal_simdgroup_source(&op, &buffers, &row, &col)
    } else {
        scalar_source(&op, dialect, &buffers, &row, &col)
    }
}

fn metal_simdgroup_source(op: &TileOperation, buffers: &[String; 3], row: &str, col: &str) -> String {
    let initial = if op.accumulate {
        format!("simdgroup_load(bp_tensor_result, {}, {}, ulong2({}, {}), false);", buffers[2], op.n, col, row)
    } else {
        "simdgroup_float8x8 bp_tensor_result = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);".into()
    };
    let result_decl = if op.accumulate { "simdgroup_float8x8 bp_tensor_result;\n" } else { "" };
    let b_load = if op.transpose_b {
        format!("simdgroup_load(bp_tensor_b, {}, {}, ulong2(bp_tensor_k, {}), true);", buffers[1], op.k, col)
    } else {
        format!("simdgroup_load(bp_tensor_b, {}, {}, ulong2({}, bp_tensor_k), false);", buffers[1], op.n, col)
    };
    format!(
        "{{\n{result_decl}{initial}\nsimdgroup_float8x8 bp_tensor_a;\nsimdgroup_float8x8 bp_tensor_b;\nfor (uint bp_tensor_k = 0; bp_tensor_k < {k}; bp_tensor_k += 8) {{\n    simdgroup_load(bp_tensor_a, {a}, {k}, ulong2(bp_tensor_k, {row}), false);\n    {b_load}\n    simdgroup_multiply_accumulate(bp_tensor_result, bp_tensor_a, bp_tensor_b, bp_tensor_result);\n}}\nsimdgroup_store(bp_tensor_result, {c}, {n}, ulong2({col}, {row}), false);\n}}",
        k = op.k, n = op.n, a = buffers[0], c = buffers[2],
    )
}

/// WMMA TF32 tile (opt-in): each warp owns whole 16x16 output fragments and
/// reduces over K in steps of 8. Eligibility that depends only on the static
/// shape is decided here; origin alignment/bounds are uniform runtime checks
/// that fall back to the exact fp32 scalar tile. Inputs are rounded to TF32.
fn cuda_tf32_source(op: &TileOperation, buffers: &[String; 3], row: &str, col: &str) -> String {
    let scalar = scalar_source(op, Dialect::Cuda, buffers, row, col);
    if !op.rows.is_multiple_of(16) || !op.cols.is_multiple_of(16)
        || !op.k.is_multiple_of(8) || !op.n.is_multiple_of(8)
    {
        return scalar;
    }
    let (a, b, c) = (&buffers[0], &buffers[1], &buffers[2]);
    let (b_layout, b_ld) = if op.transpose_b { ("col_major", op.k) } else { ("row_major", op.n) };
    let b_index = if op.transpose_b {
        "(bp_wmma_col + bp_wmma_fc * 16) * {k}".replace("{k}", &op.k.to_string())
    } else {
        "bp_wmma_col + bp_wmma_fc * 16".to_string()
    };
    let b_step = if op.transpose_b { "bp_wmma_k".to_string() } else { format!("bp_wmma_k * {}", op.n) };
    let init = if op.accumulate {
        format!("nvcuda::wmma::load_matrix_sync(bp_wmma_c, {c} + (bp_wmma_row + bp_wmma_fr * 16) * {n} + bp_wmma_col + bp_wmma_fc * 16, {n}, nvcuda::wmma::mem_row_major);", n = op.n)
    } else {
        "nvcuda::wmma::fill_fragment(bp_wmma_c, 0.0f);".to_string()
    };
    format!(
        r#"if (({row}) >= 0 && ({col}) >= 0 && (({row}) + {rows}) <= {m} && (({col}) + {cols}) <= {n} && (({col}) % 8) == 0 && ((int64_t)blockDim.x * (int64_t)blockDim.y * (int64_t)blockDim.z) % 32 == 0) {{
const int64_t bp_wmma_row = {row};
const int64_t bp_wmma_col = {col};
const int64_t bp_wmma_threads = (int64_t)blockDim.x * (int64_t)blockDim.y * (int64_t)blockDim.z;
const int64_t bp_wmma_lane = (int64_t)threadIdx.x + (int64_t)blockDim.x * ((int64_t)threadIdx.y + (int64_t)blockDim.y * (int64_t)threadIdx.z);
__syncthreads();
for (int64_t bp_wmma_f = bp_wmma_lane / 32; bp_wmma_f < {frags}; bp_wmma_f += bp_wmma_threads / 32) {{
    const int64_t bp_wmma_fr = bp_wmma_f / {fcols};
    const int64_t bp_wmma_fc = bp_wmma_f % {fcols};
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> bp_wmma_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::{b_layout}> bp_wmma_b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> bp_wmma_c;
    {init}
    for (int64_t bp_wmma_k = 0; bp_wmma_k < {k}; bp_wmma_k += 8) {{
        nvcuda::wmma::load_matrix_sync(bp_wmma_a, {a} + (bp_wmma_row + bp_wmma_fr * 16) * {k} + bp_wmma_k, {k});
        nvcuda::wmma::load_matrix_sync(bp_wmma_b, {b} + {b_step} + {b_index}, {b_ld});
        for (int bp_wmma_i = 0; bp_wmma_i < bp_wmma_a.num_elements; ++bp_wmma_i) bp_wmma_a.x[bp_wmma_i] = nvcuda::wmma::__float_to_tf32(bp_wmma_a.x[bp_wmma_i]);
        for (int bp_wmma_i = 0; bp_wmma_i < bp_wmma_b.num_elements; ++bp_wmma_i) bp_wmma_b.x[bp_wmma_i] = nvcuda::wmma::__float_to_tf32(bp_wmma_b.x[bp_wmma_i]);
        nvcuda::wmma::mma_sync(bp_wmma_c, bp_wmma_a, bp_wmma_b, bp_wmma_c);
    }}
    nvcuda::wmma::store_matrix_sync({c} + (bp_wmma_row + bp_wmma_fr * 16) * {n} + bp_wmma_col + bp_wmma_fc * 16, bp_wmma_c, {n}, nvcuda::wmma::mem_row_major);
}}
__syncthreads();
}} else {{
{scalar}
}}"#,
        m = op.m, n = op.n, k = op.k, rows = op.rows, cols = op.cols,
        frags = (op.rows / 16) * (op.cols / 16), fcols = op.cols / 16,
    )
}

fn scalar_source(
    op: &TileOperation,
    dialect: Dialect,
    buffers: &[String; 3],
    row: &str,
    col: &str,
) -> String {
    let wgsl = matches!(dialect, Dialect::Wgsl);
    // All temporaries live in a new scope. Avoid hiding operands or origins.
    let mut prefix = "bp_tensor_".to_string();
    while buffers.iter().any(|v| v.contains(&prefix))
        || row.contains(&prefix)
        || col.contains(&prefix)
    {
        prefix.push('x');
    }
    let v = |name: &str| format!("{prefix}{name}");
    let (thread, dim, barrier) = match dialect {
        Dialect::Cuda | Dialect::CudaTf32 | Dialect::Rocm => ("threadIdx", "blockDim", "__syncthreads();"),
        Dialect::Metal | Dialect::MetalNative => (
            "__thread_pos",
            "__block_dim",
            "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);",
        ),
        Dialect::Wgsl => (
            "bp_tid",
            "bp_bdim",
            "storageBarrier();\nworkgroupBarrier();",
        ),
    };
    let declaration = |name: &str, value: String| {
        if wgsl {
            format!("let {}: i32 = {};", v(name), value)
        } else {
            format!("const int64_t {} = {};", v(name), value)
        }
    };
    let component = |base: &str, axis: &str| {
        if wgsl {
            format!("i32({base}.{axis})")
        } else {
            format!("(int64_t){base}.{axis}")
        }
    };
    let lane = format!(
        "{} + {} * ({} + {} * {})",
        component(thread, "x"),
        component(dim, "x"),
        component(thread, "y"),
        component(dim, "y"),
        component(thread, "z")
    );
    let stride = format!(
        "{} * {} * {}",
        component(dim, "x"),
        component(dim, "y"),
        component(dim, "z")
    );
    let loop_head = |name: &str, start: &str, limit: usize, step: &str| {
        if wgsl {
            format!(
                "for (var {}: i32 = {}; {} < {}; {} += {}) {{",
                v(name),
                start,
                v(name),
                limit,
                v(name),
                step
            )
        } else {
            format!(
                "for (int64_t {} = {}; {} < {}; {} += {}) {{",
                v(name),
                start,
                v(name),
                limit,
                v(name),
                step
            )
        }
    };
    let access = |buffer: &str, index: String| {
        if wgsl {
            format!("{buffer}[u32({index})]")
        } else {
            format!("{buffer}[{index}]")
        }
    };
    let dst = access(
        &buffers[2],
        format!("{} * {} + {}", v("row"), op.n, v("col")),
    );
    let left = access(&buffers[0], format!("{} * {} + {}", v("row"), op.k, v("k")));
    let right = if op.transpose_b {
        access(&buffers[1], format!("{} * {} + {}", v("col"), op.k, v("k")))
    } else {
        access(&buffers[1], format!("{} * {} + {}", v("k"), op.n, v("col")))
    };
    let initial = if op.accumulate {
        dst.clone()
    } else {
        "0.0".into()
    };
    let mut lines = vec![
        "{".into(),
        barrier.into(),
        declaration("rowOrigin", row.into()),
        declaration("colOrigin", col.into()),
        declaration("lane", lane),
        declaration("stride", stride),
    ];
    // Guard origins before adding offsets: out-of-range origins do no work and
    // subtraction avoids overflow in origin + tileOffset at matrix boundaries.
    lines.push(format!(
        "if ({} >= 0 && {} < {} && {} >= 0 && {} < {}) {{",
        v("rowOrigin"),
        v("rowOrigin"),
        op.m,
        v("colOrigin"),
        v("colOrigin"),
        op.n
    ));
    lines.push(loop_head(
        "cell",
        &v("lane"),
        op.rows * op.cols,
        &v("stride"),
    ));
    lines.push(declaration("dr", format!("{} / {}", v("cell"), op.cols)));
    lines.push(declaration("dc", format!("{} % {}", v("cell"), op.cols)));
    lines.push(format!(
        "if ({} < {} - {} && {} < {} - {}) {{",
        v("dr"),
        op.m,
        v("rowOrigin"),
        v("dc"),
        op.n,
        v("colOrigin")
    ));
    lines.push(declaration(
        "row",
        format!("{} + {}", v("rowOrigin"), v("dr")),
    ));
    lines.push(declaration(
        "col",
        format!("{} + {}", v("colOrigin"), v("dc")),
    ));
    lines.push(if wgsl {
        format!("var {}: f32 = {};", v("sum"), initial)
    } else {
        format!("float {} = {};", v("sum"), initial)
    });
    lines.push(loop_head("k", "0", op.k, "1"));
    lines.push(format!("{} += {left} * {right};", v("sum")));
    lines.push("}".into());
    lines.push(format!("{dst} = {};", v("sum")));
    lines.extend([
        "}".into(),
        "}".into(),
        "}".into(),
        barrier.into(),
        "}".into(),
    ]);
    lines.join("\n")
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    fn operation() -> TileOperation {
        let source = "kernel Matrix:\n    let [float32, k = 5, m = 3]'global a\n    let [float32, n = 7, k = 5]'global b\n    mut [float32, n = 7, m = 3]'unified c\n    def ():\n        .tensors.matmulTile(a, b, c, row = 0, col = 0, rows = 2, cols = 4)\n";
        let program = crate::parser::parse(crate::lexer::lex(source).unwrap()).unwrap();
        let Item::Kernel(kernel) = &program.items[0] else {
            panic!()
        };
        let Stmt::Expr(call) = &kernel.methods[0].body[0] else {
            panic!()
        };
        crate::checker::tensor::resolve(call, kernel).unwrap()
    }

    #[test]
    fn tensor_fallback_memory_order_and_hygiene() {
        let op = operation();
        for (dialect, barrier) in [
            (Dialect::Cuda, "__syncthreads();"),
            (Dialect::Rocm, "__syncthreads();"),
            (
                Dialect::Metal,
                "threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);",
            ),
            (Dialect::Wgsl, "storageBarrier();\nworkgroupBarrier();"),
        ] {
            let source = scalar_source(
                &op,
                dialect,
                &["a".into(), "b".into(), "bp_tensor_sum".into()],
                "0",
                "0",
            );
            assert_eq!(source.matches(barrier).count(), 2);
            assert!(source.contains("bp_tensor_xsum"));
            assert!(source.contains("bp_tensor_sum["));
        }
    }

    #[test]
    fn linear_uses_gguf_row_major_weight_layout() {
        let mut op = operation();
        op.transpose_b = true;
        let source = scalar_source(
            &op,
            Dialect::Cuda,
            &["a".into(), "weights".into(), "c".into()],
            "0",
            "0",
        );
        assert!(source.contains("weights[bp_tensor_col * 5 + bp_tensor_k]"), "{source}");
    }

    /// Compile the emitted CUDA/HIP-style scalar body as ordinary C++, with
    /// thread indices supplied by the harness. This validates arithmetic,
    /// ownership, edge guards and accumulation, not real GPU synchronization.
    #[test]
    #[cfg(unix)]
    fn tensor_fallback_computes_rectangular_multiblock_tiles() {
        use std::{fs, process::Command};
        let mut op = operation();
        let buffers = ["a".into(), "b".into(), "c".into()];
        let matmul = scalar_source(&op, Dialect::Cuda, &buffers, "blockY * 2", "blockX * 4");
        op.accumulate = true;
        let mma = scalar_source(&op, Dialect::Rocm, &buffers, "blockY * 2", "blockX * 4");
        let source = format!(
            r#"
#include <cstdint>
struct Dim {{ int x,y,z; }};
Dim threadIdx, blockDim;
void __syncthreads() {{}}
float a[15], b[35], c[23];
void multiply(int blockX, int blockY) {{ {matmul} }}
void accumulate(int blockX, int blockY) {{ {mma} }}
int main() {{
  for (int i=0;i<15;++i) a[i] = float(i % 7 - 3);
  for (int i=0;i<35;++i) b[i] = float(i % 5 - 2);
  for (Dim dimensions : {{Dim{{1,1,1}}, Dim{{3,2,1}}, Dim{{4,4,2}}}}) {{
    blockDim = dimensions;
    for (int i=0;i<23;++i) c[i] = -999;
    for (int pass=0;pass<2;++pass) {{
      for (int by=0;by<3;++by) for (int bx=0;bx<3;++bx)
        for (int z=0;z<blockDim.z;++z) for (int y=0;y<blockDim.y;++y) for (int x=0;x<blockDim.x;++x) {{
          threadIdx = {{x,y,z}};
          if (pass==0) multiply(bx,by); else accumulate(bx,by);
        }}
      for (int row=0;row<3;++row) for (int col=0;col<7;++col) {{
        float expected = 0;
        for (int k=0;k<5;++k) expected += a[row*5+k]*b[k*7+col];
        if (c[row*7+col] != expected*(pass+1)) return 1;
      }}
      if (c[21] != -999 || c[22] != -999) return 2;
    }}
  }}
}}
"#
        );
        let root =
            std::env::temp_dir().join(format!("boring-tensor-fallback-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let file = root.join("fallback.cpp");
        fs::write(&file, format!("#include <initializer_list>\n{source}")).unwrap();
        let binary = root.join("fallback");
        let output = Command::new("c++")
            .args(["-std=c++11", "-O0"])
            .arg(&file)
            .arg("-o")
            .arg(&binary)
            .output()
            .expect("C++ compiler required for fallback verification");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(Command::new(&binary).status().unwrap().success());
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod wmma_tests {
    use super::*;
    use std::{fs, process::Command};

    fn operation(method: &str) -> TileOperation {
        let b_decl = if method == "linearTile" { "[float32, k = 16, n = 32]" } else { "[float32, n = 32, k = 16]" };
        let source = format!("kernel Matrix:\n    let [float32, k = 16, m = 32]'global a\n    let {b_decl}'global b\n    mut [float32, n = 32, m = 32]'unified c\n    def ():\n        .tensors.{method}(a, b, c, row = .block.y * 16, col = .block.x * 16, rows = 16, cols = 16)\n");
        let program = crate::parser::parse(crate::lexer::lex(&source).unwrap()).unwrap();
        let Item::Kernel(kernel) = &program.items[0] else { panic!() };
        let Stmt::Expr(call) = &kernel.methods[0].body[0] else { panic!() };
        crate::checker::tensor::resolve(call, kernel).unwrap()
    }

    /// Compiles the emitted WMMA TF32 tile against a scalar emulation of
    /// `nvcuda::wmma` (one emulated thread stands in for each warp),
    /// checking K/N indexing, B layout, accumulation and tile origins. It
    /// cannot validate real fragment layouts or GPU memory ordering.
    #[test]
    #[cfg(unix)]
    fn tf32_tile_matches_reference_for_matmul_mma_and_linear() {
        let buffers = ["a".into(), "b".into(), "c".into()];
        let call = |method: &str| cuda_tf32_source(&operation(method), &buffers, "blockY * 16", "blockX * 16");
        let (matmul, mma, linear) = (call("matmulTile"), call("mmaTile"), call("linearTile"));
        for body in [&matmul, &mma, &linear] {
            assert!(body.contains("mma_sync") && !body.contains("else {\nbp_tensor"), "{body}");
        }
        let source = format!(r#"
#include <cstdint>
#include <type_traits>
struct Dim3 {{ int x, y, z; }};
Dim3 threadIdx, blockDim, blockIdx;
void __syncthreads() {{}}
namespace nvcuda {{ namespace wmma {{
struct matrix_a {{}}; struct matrix_b {{}}; struct accumulator {{}};
struct row_major {{}}; struct col_major {{}}; enum layout_t {{ mem_row_major, mem_col_major }};
namespace precision {{ struct tf32 {{}}; }}
template <class U, int M, int N, int K, class T = float, class L = void>
struct fragment {{ static const int num_elements = 4; float x[4]; float v[16][16]; }};
template <int M, int N, int K, class T, class L>
void load_matrix_sync(fragment<matrix_a, M, N, K, T, L>& f, const float* p, unsigned ld) {{
  for (int i = 0; i < 16; ++i) for (int k = 0; k < 8; ++k) f.v[i][k] = p[i * ld + k];
}}
template <int M, int N, int K, class T, class L>
void load_matrix_sync(fragment<matrix_b, M, N, K, T, L>& f, const float* p, unsigned ld) {{
  for (int k = 0; k < 8; ++k) for (int j = 0; j < 16; ++j)
    f.v[k][j] = std::is_same<L, col_major>::value ? p[k + j * ld] : p[k * ld + j];
}}
template <int M, int N, int K>
void load_matrix_sync(fragment<accumulator, M, N, K, float, void>& f, const float* p, unsigned ld, layout_t) {{
  for (int i = 0; i < 16; ++i) for (int j = 0; j < 16; ++j) f.v[i][j] = p[i * ld + j];
}}
template <int M, int N, int K>
void fill_fragment(fragment<accumulator, M, N, K, float, void>& f, float v) {{
  for (int i = 0; i < 16; ++i) for (int j = 0; j < 16; ++j) f.v[i][j] = v;
}}
template <int M, int N, int K>
void store_matrix_sync(float* p, const fragment<accumulator, M, N, K, float, void>& f, unsigned ld, layout_t) {{
  for (int i = 0; i < 16; ++i) for (int j = 0; j < 16; ++j) p[i * ld + j] = f.v[i][j];
}}
inline float __float_to_tf32(float v) {{ return v; }}
template <class A, class B, class C>
void mma_sync(C& d, const A& a, const B& b, const C& c) {{
  C r = c;
  for (int i = 0; i < 16; ++i) for (int j = 0; j < 16; ++j)
    for (int k = 0; k < 8; ++k) r.v[i][j] += a.v[i][k] * b.v[k][j];
  d = r;
}}
}} }}
float a[16 * 32], b[16 * 32], c[32 * 32];
void matmul() {{ {matmul} }}
void mma() {{ {mma} }}
void linear() {{ {linear} }}
template <class F> void run(F f) {{
  blockDim = {{64, 1, 1}};
  for (int by = 0; by < 2; ++by) for (int bx = 0; bx < 2; ++bx) {{
    blockIdx = {{bx, by, 0}}; // the tile body reads blockX/blockY via these macros
    // One emulated thread per warp: a real warp executes each wmma op once, jointly.
    for (int t = 0; t < 64; t += 32) {{ threadIdx = {{t, 0, 0}}; f(); }}
  }}
}}
int main() {{
  for (int i = 0; i < 16 * 32; ++i) a[i] = float(i % 7 - 3);
  for (int i = 0; i < 16 * 32; ++i) b[i] = float(i % 5 - 2);
  // a: [m=32][k=16]; plain b: [k=16][n=32]; linear b (weight): [n=32][k=16].
  float expect[32 * 32], expect_t[32 * 32];
  for (int r = 0; r < 32; ++r) for (int col = 0; col < 32; ++col) {{
    float s = 0, t = 0;
    for (int k = 0; k < 16; ++k) {{ s += a[r * 16 + k] * b[k * 32 + col]; t += a[r * 16 + k] * b[col * 16 + k]; }}
    expect[r * 32 + col] = s; expect_t[r * 32 + col] = t;
  }}
  for (float& v : c) v = -999;
  run(matmul);
  for (int i = 0; i < 1024; ++i) if (c[i] != expect[i]) return 1;
  run(mma);
  for (int i = 0; i < 1024; ++i) if (c[i] != 2 * expect[i]) return 2;
  run(linear);
  for (int i = 0; i < 1024; ++i) if (c[i] != expect_t[i]) return 3;
}}
"#);
        let source = source.replace("blockX", "blockIdx.x").replace("blockY", "blockIdx.y");
        let root = std::env::temp_dir().join(format!("boring-tf32-tile-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let file = root.join("tile.cpp");
        fs::write(&file, source).unwrap();
        let binary = root.join("tile");
        let output = Command::new("c++")
            .args(["-std=c++14", "-O0"]).arg(&file).arg("-o").arg(&binary)
            .output().expect("C++ compiler required for WMMA indexing verification");
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let status = Command::new(&binary).status().unwrap();
        assert!(status.success(), "tile reference mismatch (exit {:?})", status.code());
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
pub(crate) fn test_program() -> Program {
    crate::parser::parse(crate::lexer::lex("kernel Matrix:\n    let [Float32, k = 5, m = 3]'global a\n    let [Float32, n = 7, k = 5]'global b\n    mut [Float32, n = 7, m = 3]'unified c\n    def ():\n        .tensors.matmulTile(a, b, c, row = .block.y * 2, col = .block.x * 4, rows = 2, cols = 4)\n").unwrap()).unwrap()
}
