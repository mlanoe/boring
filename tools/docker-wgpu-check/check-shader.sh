#!/usr/bin/env bash
set -euo pipefail

# Validates (and optionally runs) a boring-generated wgpu project's WGSL
# shaders. Mount the generated <stem>_wgpu project at /work (see run.ps1).
#
# MODE=validate (default): naga parses+type-checks each shaders/*.wgsl file --
#   the same front-end wgpu itself uses at runtime -- with no Vulkan, no GPU,
#   and no device involved at all. Fast, always available, no network needed.
# MODE=run: builds and runs the full generated project for real, with wgpu's
#   Vulkan backend forced (WGPU_BACKEND=vulkan) against Mesa's lavapipe
#   software Vulkan device -- genuine compute dispatch execution on the CPU,
#   not just static validation. Slower (full cargo build) and needs network
#   access for crates.io on the first build.

MODE="${MODE:-validate}"
SHADER_GLOB="${SHADER_GLOB:-shaders/*.wgsl}"

shopt -s nullglob
shaders=($SHADER_GLOB)
shopt -u nullglob

if [ ${#shaders[@]} -eq 0 ]; then
    echo "error: no shaders matching $SHADER_GLOB in $(pwd) -- mount the generated <stem>_wgpu project at /work" >&2
    exit 1
fi

echo "== naga-cli (wgpu's own WGSL front end) =="

status=0
for f in "${shaders[@]}"; do
    echo "--- $f ---"
    if ! naga "$f"; then
        echo "$f: naga FAILED"
        status=1
        continue
    fi
    echo "$f: OK"
done

if [ "$MODE" = "run" ]; then
    if [ $status -ne 0 ]; then
        echo "skipping run: shader validation failed" >&2
        exit $status
    fi
    echo "--- cargo run --release (WGPU_BACKEND=vulkan, lavapipe) ---"
    vulkaninfo --summary || echo "warning: vulkaninfo unavailable (lavapipe not registered?)"
    if ! WGPU_BACKEND=vulkan cargo run --release; then
        echo "cargo run FAILED"
        status=1
    fi
fi

exit $status
