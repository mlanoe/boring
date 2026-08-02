# Runs check-kernel.sh against a boring-generated CUDA project via the real
# nvcc/ptxas, no NVIDIA GPU required (device-code compilation only, never
# execution). Requires Docker Desktop (WSL2 backend) on this Windows machine.
#
# Usage:
#   .\run.ps1 -ProjectDir C:\path\to\examples\vector_add_gpu_cuda
#   .\run.ps1 -ProjectDir C:\path\to\examples\vector_add_gpu_cuda -Archs "80 90"

param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectDir,

    [string]$Archs = "70 75 80 86 89 90"
)

$ErrorActionPreference = "Stop"
$ImageTag = "boring-nvcc-check"

docker build -t $ImageTag $PSScriptRoot

$resolved = (Resolve-Path $ProjectDir).Path
docker run --rm -v "${resolved}:/work" -e ARCHS="$Archs" $ImageTag
