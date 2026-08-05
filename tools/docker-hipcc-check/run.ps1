# Runs check-kernel.sh against a boring-generated ROCm project via the real
# hipcc device-code compiler, no AMD GPU required (device-code compilation
# only, never execution). Requires Docker Desktop (WSL2 backend) on this
# Windows machine.
#
# Usage:
#   .\run.ps1 -ProjectDir C:\path\to\examples\vector_add_gpu_rocm
#   .\run.ps1 -ProjectDir C:\path\to\examples\vector_add_gpu_rocm -GfxArchs "gfx1100 gfx942"

param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectDir,

    [string]$GfxArchs = "gfx1030 gfx1100 gfx1101 gfx90a gfx942"
)

$ErrorActionPreference = "Stop"
$ImageTag = "boring-hipcc-check"

docker build -t $ImageTag $PSScriptRoot

$resolved = (Resolve-Path $ProjectDir).Path
docker run --rm -v "${resolved}:/work" -e GFX_ARCHS="$GfxArchs" $ImageTag
