# Runs check-shader.sh against a boring-generated wgpu project. Default mode
# validates shaders/*.wgsl with naga -- no GPU, no Vulkan, no device at all.
# Pass -Mode run to actually build and execute the project against Mesa's
# lavapipe software Vulkan device (real execution, slower, needs network for
# the first cargo build). Requires Docker Desktop (WSL2 backend) on this
# Windows machine.
#
# Usage:
#   .\run.ps1 -ProjectDir C:\path\to\examples\vector_add_gpu_wgpu
#   .\run.ps1 -ProjectDir C:\path\to\examples\vector_add_gpu_wgpu -Mode run

param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectDir,

    [ValidateSet("validate", "run")]
    [string]$Mode = "validate"
)

$ErrorActionPreference = "Stop"
$ImageTag = "boring-wgpu-check"

docker build -t $ImageTag $PSScriptRoot

$resolved = (Resolve-Path $ProjectDir).Path
docker run --rm -v "${resolved}:/work" -e MODE="$Mode" $ImageTag
