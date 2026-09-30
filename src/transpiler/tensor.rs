//! Scalar tile fallback shared by device emitters. Public execution remains
//! gated by the checker until launch and interpreter integration is complete.
use crate::{
    ast::*,
    checker::tensor::{resolve_fields, TileOperation},
};

#[derive(Clone, Copy)]
pub(super) enum Dialect {
    Cuda,
    Rocm,
    Metal,
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
    scalar_source(&op, dialect, &buffers, &row, &col)
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
        Dialect::Cuda | Dialect::Rocm => ("threadIdx", "blockDim", "__syncthreads();"),
        Dialect::Metal => (
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
mod tests {
    use super::*;
    fn operation() -> TileOperation {
        let source = "kernel Matrix:\n    let [float32, k = 5, m = 3]'global a\n    let [float32, n = 7, k = 5]'global b\n    mut [float32, n = 7, m = 3]'unified c\n    def ():\n        gpu.tensor.matmulTile(a, b, c, row = 0, col = 0, rows = 2, cols = 4)\n";
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
pub(crate) fn test_program() -> Program {
    crate::parser::parse(crate::lexer::lex("kernel Matrix:\n    let [Float32, k = 5, m = 3]'global a\n    let [Float32, n = 7, k = 5]'global b\n    mut [Float32, n = 7, m = 3]'unified c\n    def ():\n        gpu.tensor.matmulTile(a, b, c, row = gpu.block.y * 2, col = gpu.block.x * 4, rows = 2, cols = 4)\n").unwrap()).unwrap()
}
