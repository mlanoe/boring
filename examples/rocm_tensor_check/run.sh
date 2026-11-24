#!/usr/bin/env bash
# Builds examples/rocm_tensor_check for ROCm and runs it with the default portable kernels,
# then with the opt-in rocBLAS path. Needs a ROCm install (hipcc, librocblas) and an AMD GPU.
# Usage: examples/rocm_tensor_check/run.sh [path-to-boring-binary]
set -euo pipefail
BORING="${1:-boring}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
# gfx1032 (RX 6600) has no rocBLAS Tensile library of its own; reuse the gfx1030 one.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"

run() {
    local label="$1" toml="$2"
    printf '%s' "$toml" > boring.toml
    "$BORING" build --target rocm main.br >/dev/null
    echo "== $label =="
    (cd main_rocm && cargo run --release --quiet 2>&1 | grep -E "max abs error|panicked|Error|error:" || true)
}

run "default (portable kernels)" $'[package]\nname = "rocm_tensor_check"\n'
run "rocblas (opt-in)" $'[package]\nname = "rocm_tensor_check"\n\n[tensor.matmul]\nalgorithm = "rocblas"\n'
rm -f boring.toml
