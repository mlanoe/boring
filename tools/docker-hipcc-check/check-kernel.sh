#!/usr/bin/env bash
set -euo pipefail

# Compiles a boring-generated ROCm project's kernels/main.hip with the real
# hipcc device-code compiler, targeting a set of AMD GPU architectures --
# broader hardware coverage than any single physical GPU, and entirely
# without one. Mount the generated <stem>_rocm project at /work (see run.ps1).

HIP_FILE="${HIP_FILE:-kernels/main.hip}"
# gfx1030=RX6000(RDNA2) gfx1100/1101=RX7000(RDNA3) gfx90a=MI200(CDNA2) gfx942=MI300(CDNA3)
GFX_ARCHS="${GFX_ARCHS:-gfx1030 gfx1100 gfx1101 gfx90a gfx942}"

if [ ! -f "$HIP_FILE" ]; then
    echo "error: $HIP_FILE not found in $(pwd) -- mount the generated <stem>_rocm project at /work" >&2
    exit 1
fi

echo "== $(hipcc --version | head -1) =="

status=0
for arch in $GFX_ARCHS; do
    co="/tmp/out_${arch}.hipfb"
    echo "--- ${arch} ---"

    # --genco ("generate code object") produces a loadable fat-binary code
    # object, the HIP analogue of `nvcc --ptx` -- never touches a driver or
    # a device, only the offline AMDGPU LLVM backend for that target.
    if ! hipcc --genco -O2 --offload-arch="${arch}" -o "$co" "$HIP_FILE"; then
        echo "${arch}: hipcc FAILED"
        status=1
        continue
    fi
    echo "${arch}: OK"
done

exit $status
