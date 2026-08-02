#!/usr/bin/env bash
set -euo pipefail

# Compiles a boring-generated CUDA project's kernels/main.cu with the real
# nvcc + ptxas, targeting every real GPU architecture in gpu-profiles/ plus
# Colab's T4 (sm_75) -- broader hardware coverage than any single physical
# GPU, and entirely without one. Mount the generated <stem>_cuda project at
# /work (see run.ps1).

CU_FILE="${CU_FILE:-kernels/main.cu}"
# sm_70=v100 sm_75=t4(colab) sm_80=a100 sm_86=rtx3090/default sm_89=rtx4090 sm_90=h100
ARCHS="${ARCHS:-70 75 80 86 89 90}"

if [ ! -f "$CU_FILE" ]; then
    echo "error: $CU_FILE not found in $(pwd) -- mount the generated <stem>_cuda project at /work" >&2
    exit 1
fi

echo "== $(nvcc --version | grep release) =="

status=0
for arch in $ARCHS; do
    ptx="/tmp/out_sm${arch}.ptx"
    cubin="/tmp/out_sm${arch}.cubin"
    echo "--- sm_${arch} ---"

    if ! nvcc -arch="sm_${arch}" --ptx "$CU_FILE" -o "$ptx"; then
        echo "sm_${arch}: nvcc FAILED"
        status=1
        continue
    fi
    if ! ptxas -arch="sm_${arch}" "$ptx" -o "$cubin"; then
        echo "sm_${arch}: ptxas FAILED"
        status=1
        continue
    fi
    echo "sm_${arch}: OK"
done

exit $status
